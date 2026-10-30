"""Bounded direct generation calls; full prompts and responses stay in the run."""
import json
import math
import re
import time
import threading
import asyncio
from dataclasses import dataclass
import httpx
from .common import ROOT, write, event, check_time
from .provider_errors import response_error
from t4_splitter import count_tokens


class BudgetError(RuntimeError): pass


@dataclass(frozen=True)
class RawUserContent:
    """Explicit already-serialized user content for controlled wire experiments."""
    content: str

    def __post_init__(self):
        if not isinstance(self.content,str):raise TypeError('raw user content must be str')


def transient_proxy_status(exc):
    """Only explicit temporary CONNECT gateway failures, never proxy auth."""
    if not isinstance(exc,httpx.ProxyError):return None
    match=re.match(r'^(502|503|504)(?:\s|$)',str(exc).strip())
    return int(match[1]) if match else None


def sampling_parameters(config,stage):
    params={'temperature':.1}
    if stage!='predict':return params
    for field,upper in (('temperature',2),('top_p',1)):
        value=config.get('prediction_'+field)
        if value is None:continue
        if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=upper or (field=='top_p' and value==0):
            raise ValueError('invalid_prediction_'+field)
        params[field]=value
    return params


def thinking_enabled(config,stage):
    """Allow a bounded planner-only reasoning treatment without changing later stages."""
    return bool(config.get('enable_thinking',False) or
                (stage=='query' and config.get('query_enable_thinking',False)))


def input_messages(system, payload):
    user_content=payload.content if isinstance(payload,RawUserContent) else json.dumps(payload,ensure_ascii=False)
    return [{'role':'system','content':system},
            {'role':'user','content':user_content}]


def input_tokens(system, payload):
    return count_tokens(json.dumps(input_messages(system,payload),ensure_ascii=False))+64


def decode_model_object(content):
    """Read one complete object; never recover a partial or alternative answer.

    A stopped model occasionally appends one or two unmatched closing brackets
    after an otherwise complete JSON object. Preserve the raw response and audit
    this purely syntactic normalization. Trailing text/objects still fail.
    """
    normalizations=[];text=content.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text=text[8:-4].strip();normalizations.append('outer_json_code_fence')
    def unique_keys(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('duplicate_json_key: '+key)
            out[key]=value
        return out
    decoder=json.JSONDecoder(object_pairs_hook=unique_keys)
    data,end=decoder.raw_decode(text)
    suffix=text[end:].strip()
    if suffix:
        if isinstance(data,dict) and re.fullmatch(r'[}\]]{1,2}',suffix):
            normalizations.append('removed_unmatched_trailing_closers:'+suffix)
        else:raise ValueError('trailing_content_after_json_object')
    if not isinstance(data,dict):raise ValueError('expected_json_object')
    return data,normalizations


class Model:
    def __init__(self, config, output, deadline):
        self.config,self.output,self.deadline=config,output,deadline
        self.requests=0;self.input_tokens=0;self.output_tokens=0
        self.max_requests=config.get('max_requests',25)
        if type(self.max_requests) is not int or self.max_requests<1:raise ValueError('invalid max_requests')
        if config.get('joint_input_wire','original-json') not in ('original-json','compact-xml'):
            raise ValueError('unknown joint input wire')
        if config.get('joint_input_wire','original-json')!='original-json' and not config.get('joint_relevance_facts'):
            raise ValueError('joint input wire requires joint relevance+facts')
        self.max_input=config.get('max_input_tokens',None if config.get('provider')=='house' else 1000000)
        if self.max_input is not None and (type(self.max_input) is not int or self.max_input<1):raise ValueError('invalid max_input_tokens')
        self._lock=threading.RLock()
        self._thread=threading.local()
        self._clients=[]
        self.records=[]

    @property
    def client(self):
        return getattr(self._thread,'client',None)

    @client.setter
    def client(self,value):
        self._thread.client=value
        if value is not None:
            with self._lock:self._clients.append(value)

    def _close_current_client(self):
        client=self.client
        if client is not None:
            client.close()
            self._thread.client=None
            with self._lock:
                self._clients=[c for c in self._clients if c is not client]

    def close(self):
        """Call after workers have joined; reconnects close only their own client."""
        with self._lock:
            clients,self._clients=self._clients,[]
        for client in clients:client.close()
        self._thread.client=None

    def call(self, stage, system, payload, validator, reserve=0, request_budget=None, *, deadline=None,max_attempts=None):
        steps=self._call_steps(stage,system,payload,validator,reserve,request_budget,deadline=deadline,max_attempts=max_attempts)
        value=None;error=None
        try:
            while True:
                try:operation=steps.throw(error) if error is not None else steps.send(value)
                except StopIteration as done:return done.value
                value=None;error=None
                try:
                    if operation[0]=='post':
                        if self.client is None:
                            self.client=httpx.Client(follow_redirects=False,
                                limits=httpx.Limits(max_connections=1,max_keepalive_connections=1,keepalive_expiry=60))
                        value=self.client.post(operation[1],**operation[2])
                    elif operation[0]=='sleep':time.sleep(operation[1])
                    elif operation[0]=='reset':self._close_current_client()
                    else:raise RuntimeError('unknown_model_transport_operation')
                except BaseException as exc:error=exc
        finally:steps.close()

    async def acall(self, stage, system, payload, validator, reserve=0, request_budget=None, *, deadline=None,max_attempts=None):
        if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline)):
            raise ValueError('invalid_call_deadline')
        limit=min(self.deadline,deadline) if deadline is not None else self.deadline
        check_time(limit)
        steps=self._call_steps(stage,system,payload,validator,reserve,request_budget,deadline=limit,max_attempts=max_attempts)
        client=None;value=None;error=None
        try:
            async with asyncio.timeout(max(0,limit-time.time())):
                while True:
                    try:operation=steps.throw(error) if error is not None else steps.send(value)
                    except StopIteration as done:return done.value
                    value=None;error=None
                    try:
                        if operation[0]=='post':
                            if client is None:
                                client=httpx.AsyncClient(follow_redirects=False,
                                    limits=httpx.Limits(max_connections=1,max_keepalive_connections=1,keepalive_expiry=60))
                            value=await client.post(operation[1],**operation[2])
                        elif operation[0]=='sleep':await asyncio.sleep(operation[1])
                        elif operation[0]=='reset':
                            if client is not None:await client.aclose();client=None
                        else:raise RuntimeError('unknown_model_transport_operation')
                    except BaseException as exc:error=exc
        finally:
            steps.close()
            if client is not None:await client.aclose()

    def _call_steps(self, stage, system, payload, validator, reserve=0, request_budget=None, *, deadline=None,max_attempts=None):
        if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline)):
            raise ValueError('invalid_call_deadline')
        captured_limit=min(self.deadline,deadline) if deadline is not None else self.deadline
        # No shared deadline mutation. A later task-level shortening still wins.
        def current_limit():return min(self.deadline,captured_limit)
        wire_audit=None
        joint_wire=self.config.get('joint_input_wire','original-json')
        if stage=='facts' and joint_wire!='original-json':
            from .joint_input_wire import apply
            system,payload,wire_audit=apply(system,payload,joint_wire)
        messages=input_messages(system,payload)
        repair=0; transport_failures=0
        budget_retry = stage == 'predict' and self.config.get('prediction_budget_retry', False)
        attempts=self.max_requests if budget_retry else 4
        if max_attempts is not None:
            if type(max_attempts) is not int or max_attempts<1:raise ValueError('invalid max_attempts')
            attempts=min(attempts,max_attempts)
        for attempt in range(attempts):
            check_time(current_limit())
            serialized=json.dumps(messages,ensure_ascii=False)
            estimated=count_tokens(serialized)+64
            # UTF-8 bytes provide a conservative admission bound when provider
            # tokenizer is unavailable. Actual provider usage replaces it below.
            admission=max(estimated,len(serialized.encode('utf-8'))+128)
            if estimated+4000>self.config.get('context_window',32000):raise BudgetError('context_budget: split stage payload')
            with self._lock:
                check_time(current_limit())
                reserved=max(reserve,request_budget.reserve()) if request_budget is not None else reserve
                if self.requests>=self.max_requests-reserved:raise BudgetError('request_budget_reserved_for_prediction')
                input_reserve=request_budget.input_reserve() if request_budget is not None and hasattr(request_budget,'input_reserve') else 0
                if self.max_input is not None and self.input_tokens+admission+input_reserve>self.max_input:raise BudgetError('input_token_budget_reserved_for_prediction' if input_reserve else 'input_token_budget')
                self.requests+=1;self.input_tokens+=admission
                if request_budget is not None:request_budget.admitted()
                request_id=self.requests
            stem=f'{request_id:02d}-{stage}'
            request={'model':self.config['model'],'messages':messages,
                     'response_format':{'type':'json_object'},**sampling_parameters(self.config,stage)}
            request['max_tokens']=(self.config.get('query_max_tokens')
                                   if stage=='query' and self.config.get('query_max_tokens') is not None
                                   else self.config.get('max_tokens',4000))
            if stage=='predict' and self.config.get('prediction_candidates',1)>1:
                request['n']=self.config['prediction_candidates']
            stage_thinking=thinking_enabled(self.config,stage)
            if self.config.get('provider')=='house':
                request['chat_template_kwargs']={'enable_thinking':stage_thinking}
            elif self.config.get('provider')=='aliyun':
                request['enable_thinking']=stage_thinking
            else:
                request['reasoning']={'enabled':stage_thinking}
                # Local paired experiment only. House remains independently
                # configured; effort is not a guaranteed token-budget ceiling.
                if stage=='predict' and self.config.get('prediction_reasoning_effort'):
                    effort=self.config['prediction_reasoning_effort']
                    if effort not in ('minimal','low'):raise ValueError('Experimental prediction effort must be minimal or low')
                    request['reasoning']={'effort':effort,'exclude':self.config.get('prediction_reasoning_exclude',True)}
                if self.config.get('openrouter_sort'):
                    if self.config['openrouter_sort'] not in ('price','throughput','latency'):raise ValueError('Invalid OpenRouter provider sorting')
                    request['provider']={'sort':self.config['openrouter_sort']}
                if self.config.get('openrouter_provider'):
                    request['provider']={'only':[self.config['openrouter_provider']],'allow_fallbacks':False}
            artifact={'request':request,'estimated_input_tokens':estimated,'reserved_input_tokens':admission}
            if wire_audit is not None:artifact['input_wire_audit']=wire_audit
            write(self.output/'model'/f'{stem}-request.json',artifact)
            start=time.monotonic();record={'stage':stage,'request':request_id,'repair':repair,
                                          'enable_thinking':stage_thinking,
                     'transport_attempt':transport_failures+1}
            content=None
            try:
                check_time(current_limit())
                response=yield ('post',self.config['base_url'].rstrip('/')+'/chat/completions',
                    {'headers':{'Authorization':'Bearer '+self.config['api_key']},
                     'json':request,'timeout':min(300,max(.001,current_limit()-time.time()))})
                record['http_status']=response.status_code
                if response.status_code!=200:
                    # Response error fields only; never log Authorization or keys.
                    try:
                        error=response.json().get('error',{})
                        if isinstance(error,dict):
                            record['http_error']={k:str(error[k])[:400] for k in ('code','type','message') if k in error}
                    except (ValueError,AttributeError):pass
                if response.status_code in (408,429,500,502,503,504) and (budget_retry or transport_failures<2):
                    delay=(prediction_retry_delay(response.headers,transport_failures,response.status_code)
                           if budget_retry else retry_delay(response.headers,transport_failures,response.status_code))
                    if budget_retry:self._check_prediction_retry(delay, reserve, admission, deadline=current_limit())
                    record.update(status='retryable_http',retry_delay_seconds=delay)
                    if current_limit()-time.time()<=delay+5:raise TimeoutError('deadline_before_http_retry')
                    transport_failures+=1;yield ('sleep',delay);continue
                if response.status_code!=200:raise RuntimeError('generation_http_'+str(response.status_code))
                body=response.json();write(self.output/'model'/f'{stem}-response.json',body)
                usage=body.get('usage') or {};actual=usage.get('prompt_tokens')
                with self._lock:
                    if isinstance(actual,int) and actual>=0:self.input_tokens+=actual-admission
                    self.output_tokens+=usage.get('completion_tokens') or 0
                    over_budget=self.max_input is not None and self.input_tokens>self.max_input
                record['usage']=usage
                if over_budget:raise BudgetError('provider_input_budget_exceeded')
                check_time(current_limit())
                provider_error=response_error(body)
                if provider_error is not None:
                    record['provider_error']=provider_error
                    if not provider_error['retryable'] or (not budget_retry and transport_failures>=2):
                        raise RuntimeError('generation_provider_error')
                    status=provider_error['code']
                    delay=(prediction_retry_delay(response.headers,transport_failures,status)
                           if budget_retry else retry_delay(response.headers,transport_failures,status))
                    if budget_retry:self._check_prediction_retry(delay,reserve,admission,deadline=current_limit())
                    record.update(status='retryable_provider',retry_delay_seconds=delay)
                    if current_limit()-time.time()<=delay+5:raise TimeoutError('deadline_before_provider_retry')
                    transport_failures+=1
                    yield ('sleep',delay)
                    continue
                choices=body.get('choices')
                if not isinstance(choices,list) or not choices:raise ValueError('missing_choices')
                if stage=='predict' and self.config.get('prediction_candidates',1)>1:
                    candidates=[]
                    for candidate in choices:
                        if candidate.get('finish_reason')!='stop':continue
                        text=candidate.get('message',{}).get('content')
                        if not isinstance(text,str) or not text.strip():continue
                        parsed,_=decode_model_object(text)
                        candidates.append(validator(parsed))
                    if not candidates:raise ValueError('no_valid_prediction_candidates')
                    target_type=payload.get('task',{}).get('target',{}).get('type')
                    if target_type=='regression' and len(candidates)>1:
                        import statistics
                        merged=[]
                        for entity_id in candidates[0]:
                            rows=[row for result in candidates for row in result if row.get('entity_id')==entity_id.get('entity_id')]
                            median=statistics.median(float(row['point_forecast']) for row in rows)
                            row=dict(min(rows,key=lambda x:abs(float(x['point_forecast'])-median)))
                            row['point_forecast']=median
                            row['interval']={'lo':min(float(x['interval']['lo']) for x in rows),
                                             'hi':max(float(x['interval']['hi']) for x in rows)}
                            merged.append(row)
                        result=merged
                    else:
                        result=candidates[0]
                    record['candidate_count']=len(candidates)
                    record['finish_reason']='stop'
                    record['status']='success'
                    return result
                choice=choices[0]
                record['finish_reason']=choice.get('finish_reason')
                message=choice.get('message',{})
                reasoning_fields={key:message[key] for key in ('reasoning','reasoning_content','reasoning_details')
                                  if message.get(key)}
                if reasoning_fields:
                    write(self.output/'model'/f'{stem}-reasoning.json',{
                        'stage':stage,'request':request_id,'model':body.get('model'),
                        'provider':body.get('provider'),'reasoning_fields':reasoning_fields})
                record['reasoning_content_characters']=len(message.get('reasoning') or message.get('reasoning_content') or '')
                if choice.get('finish_reason')!='stop':raise ValueError('incomplete_model_output')
                content=choice['message']['content']
                if not isinstance(content,str) or not content.strip():raise ValueError('missing_text_content')
                data,normalizations=decode_model_object(content)
                if normalizations:record['parser_normalizations']=normalizations
                result=validator(data)
                check_time(current_limit())
                record['status']='success'
                return result
            except (httpx.ConnectError,httpx.ReadError,httpx.RemoteProtocolError,httpx.TimeoutException,httpx.ProxyError) as exc:
                if isinstance(exc,httpx.ProxyError):
                    status=transient_proxy_status(exc)
                    if status is None:
                        record['status']='failed';record['error']=type(exc).__name__
                        raise
                    record['proxy_status']=status
                record['status']='transport_error';record['error']=type(exc).__name__
                code=getattr(exc,'transport_exit_code',None)
                if type(code) is int:record['transport_exit_code']=code
                if not budget_retry and transport_failures>=2:raise
                delay=min(30,2**transport_failures) if budget_retry else 2**transport_failures
                if budget_retry:self._check_prediction_retry(delay, reserve, admission, deadline=current_limit())
                if current_limit()-time.time()<=delay:raise TimeoutError('task_deadline_before_retry') from exc
                record['retry_delay_seconds']=delay
                transport_failures+=1
                try:
                    yield ('reset',)  # Close only this logical call's transport.
                    yield ('sleep',delay)
                except (asyncio.CancelledError,GeneratorExit) as interrupted:
                    record['interrupted_transport_error']=record.get('error')
                    record['status']='cancelled';record['error']=type(interrupted).__name__
                    raise
            except (ValueError,KeyError,TypeError) as exc:
                record['status']='invalid_output';record['error']=str(exc)[:500]
                if repair:raise
                repair+=1
                # One bounded schema repair, charged as a real request.
                messages=[messages[0],messages[1],{'role':'assistant','content':content if isinstance(content,str) else '{}'},
                          {'role':'user','content':'Correct the JSON only. Validation error: '+str(exc)[:500]}]
            except Exception as exc:
                record['status']='failed';record['error']=type(exc).__name__
                raise
            except BaseException as exc:
                record['status']='cancelled';record['error']=type(exc).__name__
                raise
            finally:
                record['seconds']=time.monotonic()-start
                with self._lock:
                    self.records.append(dict(record))
                    event(self.output/'model-requests.jsonl',record)
                    write(self.output/'budget.json',self.summary())
        raise ValueError('generation_validation_failed')

    def _check_prediction_retry(self, delay, reserve, admission, *, deadline=None):
        # Admission is checked again after sleeping. No quota/deadline is reset.
        with self._lock:
            if self.requests >= self.max_requests-reserve:
                raise BudgetError('request_budget_reserved_for_prediction')
            if self.max_input is not None and self.input_tokens+admission > self.max_input:
                raise BudgetError('input_token_budget')
            limit=min(self.deadline,deadline) if deadline is not None else self.deadline
            if limit-time.time() <= delay+10*(reserve+1):
                raise TimeoutError('deadline_before_prediction_retry')

    def summary(self):
        with self._lock:
            return {'requests':self.requests,'input_tokens':self.input_tokens,'output_tokens':self.output_tokens,
                    'max_requests':self.max_requests,'max_input_tokens':self.max_input,'max_output_per_request':4000}


def retry_delay(headers, attempt, status):
    """Honor delta/date Retry-After, otherwise back off for rate limiting."""
    from email.utils import parsedate_to_datetime
    raw=headers.get('retry-after')
    if raw:
        try:return min(60,max(1,float(raw)))
        except (ValueError,TypeError):
            try:return min(60,max(1,parsedate_to_datetime(raw).timestamp()-time.time()))
            except (ValueError,TypeError,OverflowError):pass
    return min(30,(5 if status==429 else 1)*(2**attempt))


def prediction_retry_delay(headers, attempt, status):
    """Do not shorten provider Retry-After; the caller rejects waits past budget."""
    from email.utils import parsedate_to_datetime
    raw=headers.get('retry-after')
    if raw is not None:
        try:value=float(raw)
        except (ValueError,TypeError):
            try:value=parsedate_to_datetime(raw).timestamp()-time.time()
            except (ValueError,TypeError,OverflowError):value=None
        if value is not None and math.isfinite(value):return max(1,value)
    return min(60,30*(2**attempt)) if status==429 else min(30,2**attempt)
