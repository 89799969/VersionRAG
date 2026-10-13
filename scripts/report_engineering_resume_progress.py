"""Extend the immutable offline delivery with synthetic version/socket evidence."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-dir', type=Path, required=True)
    parser.add_argument('--version-dir', type=Path, required=True)
    parser.add_argument('--load-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new delivery directory')
    prior = args.prior_dir.resolve()
    version_path = args.version_dir.resolve() / 'version-task.json'
    load_path = args.load_dir.resolve() / 'load-task.json'
    version = json.loads(version_path.read_text(encoding='utf-8'))
    load = json.loads(load_path.read_text(encoding='utf-8'))
    if version['status'] != 'passed' or load['status'] != 'passed' or not load['port_closed']:
        raise ValueError('Synthetic engineering validation did not finish successfully')
    if version['remote_calls'] != 0 or load['remote_provider_calls'] != 0:
        raise ValueError('Synthetic report cannot contain real provider calls')
    manifest = json.loads((prior / 'manifest.json').read_text(encoding='utf-8'))
    manifest.update({'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                     'schema_version': 'resume-evidence-engineering-interim-v1',
                     'prior_delivery_manifest_sha256': sha256(prior / 'manifest.json'),
                     'current_code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p)
                          for folder in ['app', 'scripts', 'tests'] for p in sorted((ROOT / folder).rglob('*.py'))},
                     'synthetic_protocol_inputs': {p.relative_to(ROOT).as_posix(): sha256(p) for p in [version_path, load_path]},
                     'status': 'INCOMPLETE'})
    runtime = json.loads((prior / 'runtime.json').read_text(encoding='utf-8'))
    runtime['synthetic_version_fact_chain'] = {'status': 'passed', 'tasks': 1, 'steps_passed': version['steps_passed'],
                                            'steps_total': version['steps_total'], 'scope': version['scope']}
    runtime['synthetic_socket_load'] = {k: load[k] for k in ('fixed_concurrency', 'planned_happy_requests', 'happy_protocol_passed',
                                         'total_http_requests', 'controlled_http_failures', 'owned_server_stopped', 'port_closed', 'scope')}
    runtime['synthetic_socket_window'] = load['runtime']
    runtime['synthetic_service_load'] = {'status': 'completed_with_local_doubles', 'requests': load['total_http_requests'],
                                       'concurrency': load['fixed_concurrency'], 'scope': load['scope']}
    runtime['version_answer_task'] = {'status': 'not_run_with_real_model', 'passed': None, 'total': None}
    metrics = json.loads((prior / 'metrics.json').read_text(encoding='utf-8'))
    metrics['engineering_protocol'] = {'version_fact_steps': {'numerator': version['steps_passed'], 'denominator': version['steps_total'],
                                           'unit': 'synthetic_steps_in_one_task', 'scope': version['scope']},
                                      'socket_happy_checks': {'numerator': load['happy_protocol_passed'],
                                           'denominator': load['planned_happy_requests'], 'unit': 'synthetic_http_requests', 'scope': load['scope']},
                                      'real_model_answer_quality': None}
    additions = f'''
## 请求账本与版本/部署工程验证

增加私有匿名任务/请求账本，关联固定知识版本、方法、模型及提示哈希、证据编号/哈希、嵌套阶段耗时和每次真实HTTP尝试。失败/重试也计入；provider未给usage时仅报告已知小计，完整总量保持null。问题正文、提示内容、凭据、响应正文和异常消息不入账本。文件默认在.cache，不上传公开仓库；单进程文件追加不等于多节点可靠遥测平台。

修复比较生成汇总缺失usage按0相加的问题。新增窗口程序明确HTTP成功不等于答案正确，任务编号不等于真实用户，缺人工审核不能推导质量。遥测写入失败会留下固定错误日志，不能悄悄视为完整观测。

实际合成版本任务：1条任务链、{version['steps_passed']}/{version['steps_total']}步协议检查通过；初版8000→新版9000→检索后发生回滚的在途答复仍9000→回滚后8000→构建失败仍8000。使用隔离真实知识数据库/索引和应用接口，但编码器与生成模型是受控替身，不是现实招生事实或真实模型成绩。

实际本机socket服务：固定并发{load['fixed_concurrency']}，{load['happy_protocol_passed']}/{load['planned_happy_requests']}条合成正常请求版本/事实/引用检查通过；另有预期422、504两条失败，总{load['total_http_requests']}条都进入账本。模型调用仍是受控本地传输，远端收费调用0。自己的服务线程已停止，临时端口复查不再接受连接。这个窗口仅说明小型合成部署/容量，不是生产时延或真实用户质量。

真实新验收、人审、真实模型版本任务与自愿试用仍未运行，完整P0—P3目标保持INCOMPLETE。answers.jsonl仍只代表新验收生成，当前为空；合成答复见独立version-task/load-task，不混入主成绩。
'''
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ('answers.jsonl', 'human-review.csv', 'retrieval.csv'):
        shutil.copyfile(prior / name, args.output_dir / name)
    (args.output_dir / 'changes.md').write_text((prior / 'changes.md').read_text(encoding='utf-8') + additions, encoding='utf-8')
    (args.output_dir / 'failures.md').write_text((prior / 'failures.md').read_text(encoding='utf-8') +
        '\n追加工程验证：合成负载中的空问题422和受控模型超时504已纳入请求窗口；失败构建保留旧版也已验证。'
        '这些是预设反例，不是新验收集真实退步。失败/重试usage缺失仍为null；真实模型与人工质量缺口未关闭。\n', encoding='utf-8')
    for name, data in [('manifest.json', manifest), ('runtime.json', runtime), ('metrics.json', metrics)]:
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'directory': str(args.output_dir), 'status': 'INCOMPLETE', 'files': len(list(args.output_dir.iterdir())),
                      'synthetic_version_task': '1 task, 5 steps', 'remote_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
