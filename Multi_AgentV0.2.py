#!/usr/bin/env python3
"""
Multi-Agent 研究协作脚本
使用 CrewAI 框架和本地 Ollama 模型，执行“研究员 + 主编”的协作任务。
"""

import os
import sys
import asyncio
import logging
import requests

from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew

# ==================== 环境配置 ====================
# 禁用 CrewAI 遥测（避免 Windows 上信号问题）
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"

# Windows 事件循环兼容性（防止异步死锁）
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 配置日志（调试用）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 服务可用性检查 ====================
def check_ollama_service(base_url: str, model: str) -> bool:
    """检查 Ollama 服务是否可用，并验证模型是否存在"""
    try:
        # 尝试访问 Ollama 的 API 端点
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code != 200:
            logger.error(f"Ollama 服务响应异常: {response.status_code}")
            return False

        models = response.json().get("data", [])   
        model_names = [m.get("id") for m in models if "id" in m]
        # 注意：模型名可能包含 "ollama/" 前缀，需实际匹配
        if any(model_name in model for model_name in model_names):
            logger.info(f"Ollama 服务正常，模型 {model} 已存在。")
            return True
        else:
            logger.warning(f"Ollama 服务正常，但未找到模型 {model}。可用模型: {model_names}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(f"无法连接到 Ollama 服务: {base_url}")
        return False
    except Exception as e:
        logger.error(f"检查 Ollama 服务时出错: {e}")
        return False


# ==================== 主程序 ====================
def main():
    """主函数，执行多 Agent 协作任务"""
    try:
        # ---------- 1. 配置 LLM ----------
        base_url = "http://localhost:11434/v1"
        model = "ollama/qwen2.5:7b"
        api_key = "ollama"

        # 可选：检查 Ollama 服务是否正常
        if not check_ollama_service(base_url, model):
            logger.error("Ollama 服务不可用，请确保 Ollama 已启动并拉取模型。")
            sys.exit(1)

        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.7  # 设置合适的创造性参数
        )
        logger.info(f"LLM 初始化成功: {model} @ {base_url}")

        # ---------- 2. 定义 Agent ----------
        
        researcher = Agent(
            role="资深研究员",
            goal="针对 {topic} 查找最新、最准确的信息并进行结构化整理",
            backstory="你是一名在科技领域深耕10年的研究员，擅长数据分析和事实核查。",
            llm=llm,
            allow_delegation=False,
            verbose=True   # 启用 Agent 内部日志
        )

        editor = Agent(
            role="主编",
            goal="将研究员提供的信息润色成吸引眼球的短视频脚本",
            backstory="你是百万粉丝博主的御用文案策划，擅长制造悬念和口语化表达。",
            llm=llm,
            allow_delegation=False,
            verbose=True
        )
        logger.info("Agent 创建成功")

        # ---------- 3. 定义任务 ----------
        task1 = Task(
            description="调研关于 '2025年AI Agent发展趋势' 的三个核心观点，并提供数据支撑。",
            agent=researcher,
            expected_output="三个核心观点的详细列表，每个观点附带数据来源说明"
        )

        task2 = Task(
            description="基于研究员的观点，编写一段300字左右的短视频口播文案，开头要有吸引力。",
            agent=editor,
            expected_output="完整的口播文案（300字左右）"
        )

        # ---------- 4. 组建 Crew 并执行 ----------
        crew = Crew(
            agents=[researcher, editor],
            tasks=[task1, task2],
            verbose=True   # 启用 Crew 执行过程日志
        )

        logger.info("开始执行 Crew 任务...")
        result = crew.kickoff(inputs={'topic': 'AI Agent 商业化落地'})

        # ---------- 5. 输出结果 ----------
        print("\n" + "="*50)
        print("执行结果:")
        print("="*50)
        print(result)
        print("="*50)

        return result

    except ImportError as e:
        logger.exception(f"缺少必要的依赖库: {e}")
        print("请确保已安装所需依赖: pip install crewai langchain-openai requests")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"执行过程中发生未预期的错误: {e}")
        print(f"错误详情: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()