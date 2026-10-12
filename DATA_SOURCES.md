# 数据与开源来源

- [DomainRAG](https://github.com/ShootingWong/DomainRAG)：提交 `f6ca1941a71ff19f6d73bd7588842bb92a6113e3`，提供原始问答和参考证据。上游仓库没有发现独立 LICENSE，未把其代码复制或声明为 MIT；交付包仅保留本项目需要的原始任务数据与来源 README。
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)：提交 `0e57de4394fdbc3133e352f0daff4b87cd7fff74`，MIT。本机保留克隆作为参考，交付包不包含完整上游仓库。本项目通过正常依赖复用 BM25s，而非调用完整 FlashRAG Pipeline。
- [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)：提供 DomainRAG 的预处理中文语料与统一测试格式；数据卡标注 CC-BY-SA-4.0，保留来源及该许可标记。原始材料权利仍需结合来源判断，数据卡声明不自动替代所有源材料权利信息。
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)、[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)：在线模型服务，未下载权重；API 不暴露实际权重修订，报告明确记录该限制。

语料 SHA256：`6b08f337ee4066af05d99bcf5bc29afca56f7e05df8b0feed5030dc34bfd52d9`。

基础映射 SHA256：`7e8cca569448885ddba55716ba22a73d9ebcfd239cb4132354fc31b079c1c81e`。

扩展映射 SHA256：`22b2397a0ccae715ce991a4cb3b4c33fcb731cff97a5dfa2441ed10ca91f36fd`。

映射属于派生评测资料，按来源ID和规范化原文包含关系对齐，不新增虚构官方标签。原始相同问题变体不作为相互独立的485题。交付包排除密钥、模型权重、虚拟环境与索引缓存，保留代码、任务资料、来源说明和本轮报告。
