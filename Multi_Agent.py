#!/usr/bin/env python3
"""
Multi-Agent 研究协作脚本 - 双后端稳定版

功能：使用 CrewAI 框架，支持 Ollama（本地）或 DeepSeek（云端）后端，
      执行“资深研究员”与“主编”的协作任务，生成短视频文案。

环境要求：
    1. 激活虚拟环境：multi_agent_env
    2. 安装依赖：MulAgrequires.txt Create_MulAg_env.ps1
    3. Ollama 需启动服务并拉取模型；DeepSeek 需提供有效 API Key

后端切换：
    修改下方 LLM_PROVIDER 变量为 "ollama" 或 "deepseek" 即可。
"""

import os
import sys
import asyncio
import logging
import requests

from crewai import Agent, Task, Crew, LLM  # 使用 CrewAI 内置 LLM 类（稳定，无需前缀问题）

# ==================== 配置区 ====================
# 选择后端: "ollama" 或 "deepseek"
LLM_PROVIDER = "deepseek"   # 在此修改

# ---------- Ollama 配置 ----------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "ollama/qwen2.5:7b"   # 必须带 "ollama/" 前缀
OLLAMA_API_KEY = "ollama"

# ---------- DeepSeek 配置 ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "openai/deepseek-chat"   # 必须带 "openai/" 前缀
DEEPSEEK_API_KEY = "sk-d2b2650965144e48849cd7d1a6147ebf"   # 请替换为真实 Key

# ---------- 通用配置 ----------
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 服务检查 ====================
def check_ollama_service(base_url: str, model_full: str) -> bool:
    """检查 Ollama 服务及模型是否存在（使用 /api/tags）"""
    actual_model = model_full.split('/')[-1]
    tags_url = f"{base_url}/api/tags"
    try:
        resp = requests.get(tags_url, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Ollama 服务响应异常: {resp.status_code}")
            return False
        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models if "name" in m]
        if any(actual_model in name for name in model_names):
            logger.info(f"Ollama 服务正常，模型 {actual_model} 已存在。")
            return True
        else:
            logger.warning(f"未找到模型 {actual_model}，可用: {model_names}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(f"无法连接到 Ollama 服务: {base_url}")
        return False
    except Exception as e:
        logger.error(f"Ollama 检查失败: {e}")
        return False

def check_deepseek_service(api_key: str, base_url: str) -> bool:
    """检查 DeepSeek API 连通性（通过 /models 端点）"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(f"{base_url}/models", headers=headers, timeout=5)
        if resp.status_code == 200:
            logger.info("DeepSeek API 服务正常。")
            return True
        else:
            logger.error(f"DeepSeek API 异常: {resp.status_code}")
            return False
    except Exception as e:
        logger.error(f"DeepSeek 检查失败: {e}")
        return False


# ==================== LLM 初始化 ====================
def init_llm():
    """根据 LLM_PROVIDER 返回对应的 CrewAI LLM 实例"""
    if LLM_PROVIDER == "ollama":
        if not check_ollama_service(OLLAMA_BASE_URL, OLLAMA_MODEL):
            logger.error("Ollama 服务不可用，请运行 'ollama serve' 并拉取模型。")
            sys.exit(1)
        llm = LLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            temperature=0.7
        )
        logger.info(f"Ollama 后端初始化成功: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        return llm

    elif LLM_PROVIDER == "deepseek":
        if not check_deepseek_service(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL):
            logger.error("DeepSeek API 不可用，请检查 API Key 和网络。")
            sys.exit(1)
        llm = LLM(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            temperature=0.7
        )
        logger.info(f"DeepSeek 后端初始化成功: {DEEPSEEK_MODEL}")
        return llm

    else:
        logger.error(f"不支持的提供商: {LLM_PROVIDER}，请选择 'ollama' 或 'deepseek'")
        sys.exit(1)


# ==================== 主程序 ====================
def main():
    try:
        llm = init_llm()

        # 定义 Agent
        researcher = Agent(
            role="资深研究员",
            goal="针对 {topic} 查找最新、最准确的信息并进行结构化整理",
            backstory="你是一名在科技领域深耕10年的研究员，擅长数据分析和事实核查。",
            llm=llm,
            allow_delegation=False,
            verbose=True
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

        # 定义任务（使用上下文依赖）
        task3 = Task(
            description="搜索 AI Agent 相关资料，为调研 '2025年AI Agent发展趋势' 提供丰富素材。",
            agent=editor,
            expected_output="一份详细的资料清单或关键信息摘要（长度不限）"
        )
        task2 = Task(
            description="参考主编提供的资料，调研关于 '2025年AI Agent发展趋势' 的三个核心观点，并提供数据支撑。",
            agent=researcher,
            expected_output="三个核心观点的详细列表，每个观点附带数据来源说明",
            context=[task3]
        )
        task1 = Task(
            description="基于研究员的研究成果，编写一段300字左右的短视频口播文案，开头要有吸引力。",
            agent=editor,
            expected_output="完整的口播文案（300字左右）",
            context=[task2]
        )

        crew = Crew(
            agents=[researcher, editor],
            tasks=[task3, task2, task1],
            verbose=True
        )

        logger.info("开始执行 Crew 任务...")
        result = crew.kickoff()

        logger.info("=" * 50)
        logger.info("执行结果:")
        logger.info("=" * 50)
        logger.info(result)
        logger.info("=" * 50)

        return result

    except ImportError as e:
        logger.exception(f"缺少依赖库: {e}")
        logger.error("请安装: pip install crewai requests")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()