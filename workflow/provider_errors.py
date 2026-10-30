"""Explicit provider failures in a nominally successful HTTP envelope."""

def response_error(body):
    if not isinstance(body,dict):return None
    choices=body.get('choices')
    choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {}
    location='top' if body.get('error') is not None else 'choice'
    error=body.get('error') if location=='top' else choice.get('error')
    if error is None and choice.get('finish_reason')!='error':return None
    error=error if isinstance(error,dict) else {}
    code=error.get('code')
    if isinstance(code,str) and code.isascii() and code.isdigit():code=int(code)
    if type(code) is not int:code=None
    metadata=error.get('metadata')
    kind=metadata.get('error_type') if isinstance(metadata,dict) else None
    transient=code in (408,429,500,502,503,504)
    if kind is not None:
        transient=transient and kind in ('request_timeout','rate_limit_exceeded','provider_unavailable','server')
    return {'location':location,'code':code,'error_type':str(kind)[:100] if kind is not None else None,
            'retryable':transient}
