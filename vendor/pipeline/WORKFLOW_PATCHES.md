# Workflow适配修改

来源：原交付的teammate_vendor；保留原O4000、表格抽取与向量节点算法。

- preprocess_unit.preprocess增加official_assignments参数。
- search_index.build_search_index接受官方关系，写入entity_ids、shared、association_source；关系参与索引identity，verify_index同步校验。
- 缺少此参数的旧CLI仍是旧推断逻辑；本workflow始终传官方关系，并在检索和引用处执行权限过滤。不要把vendor旧CLI当作新版workflow入口。

源码哈希见../../docs/implementation-source-hashes.json。
