"""
QARobot_QwenBased - 基于 Qwen 模型（通过 Ollama 本地服务）或 DeepSeek API 的对话机器人
提供带历史记忆、人设设定、历史裁剪和异常处理的聊天功能。
"""

import json
import logging
from langchain_openai import ChatOpenAI

# ==================== 头部配置：选择模型类型和参数 ====================
# 可选值: "ollama" (本地 Ollama) 或 "deepseek" (DeepSeek API)
MODEL_TYPE = "ollama"          # 修改此处即可切换模型

# Ollama 配置 (当 MODEL_TYPE = "ollama" 时生效)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"      # 本地服务不需要真实 key，但必须提供
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"   # 默认模型名，可被 ChatBot 构造函数中的 model 参数覆盖

# DeepSeek 配置 (当 MODEL_TYPE = "deepseek" 时生效)
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = "sk-d2b2650965144e48849cd7d1a6147ebf"   # 请替换为真实 API Key
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"     # 默认模型名
# ====================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatBot:
    """
    一个基于 LangChain ChatOpenAI 的聊天机器人（对接本地 Ollama 或云端 DeepSeek）。
    维护对话历史，支持人设设定和历史消息裁剪。
    """

    def __init__(self, personality: str, model: str = None, max_history_len: int = 10,
                 temperature: float = 0.7, debug: bool = True):
        """
        初始化聊天机器人。

        :param personality: 人设描述（系统提示词）
        :param model: 使用的模型名称（如果为 None，则使用默认模型）
        :param max_history_len: 保留的最近对话轮数（不包括系统提示词）
        :param temperature: 生成回复的随机性（0~1之间）
        :param debug: 是否打印详细调试信息（包括历史内容）
        """
        self.system_prompt = {"role": "system", "content": personality}
        self.history = [self.system_prompt]
        self.max_history_len = max_history_len
        self.temperature = temperature
        self.debug = debug

        # 根据 MODEL_TYPE 配置 ChatOpenAI 客户端
        if MODEL_TYPE == "ollama":
            base_url = OLLAMA_BASE_URL
            api_key = OLLAMA_API_KEY
            default_model = OLLAMA_DEFAULT_MODEL
        elif MODEL_TYPE == "deepseek":
            base_url = DEEPSEEK_BASE_URL
            api_key = DEEPSEEK_API_KEY
            default_model = DEEPSEEK_DEFAULT_MODEL
        else:
            raise ValueError(f"不支持的 MODEL_TYPE: {MODEL_TYPE}，请设置为 'ollama' 或 'deepseek'")

        # 用户未指定模型时使用默认模型
        self.model = model if model is not None else default_model

        try:
            self.client = ChatOpenAI(
                model=self.model,
                base_url=base_url,
                api_key=api_key,
                temperature=self.temperature,
                request_timeout=240.0
            )
            if self.debug:
                logger.info(f"ChatOpenAI 客户端初始化成功，模型类型: {MODEL_TYPE}，base_url: {base_url}")
        except Exception as e:
            logger.error(f"初始化 ChatOpenAI 客户端失败: {e}")
            raise

    def _trim_history(self):
        """
        裁剪对话历史，保留系统提示词和最近的 max_history_len 条消息。
        此方法会在每次用户输入后调用，确保历史长度不超过限制。
        """
        if len(self.history) > self.max_history_len + 1:
            self.history = [self.system_prompt] + self.history[-(self.max_history_len):]
            if self.debug:
                logger.debug("历史消息已裁剪，当前长度: %d", len(self.history))

    def _print_history(self):
        """打印当前对话历史（调试用），每条消息只显示前100个字符。"""
        if not self.debug:
            return
        history_lines = ["\n--- 当前对话历史 ---"]
        for i, msg in enumerate(self.history):
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            history_lines.append(f"{i+1}. {role}: {content}")
        history_lines.append("=" * 50)
        logger.info("\n".join(history_lines))

    def chat(self, user_input: str) -> str:
        """
        处理用户输入，返回机器人的回复。

        :param user_input: 用户输入的文本
        :return: 机器人的回复文本；若发生异常，返回错误提示信息
        """
        # 1. 添加用户消息到历史
        self.history.append({"role": "user", "content": user_input})
        if self.debug:
            self._print_history()

        # 2. 裁剪历史（防止超过最大长度）
        self._trim_history()
        if self.debug:
            self._print_history()

        # 3. 调用模型获取回复
        try:
            response = self.client.invoke(self.history)
            reply = response.content
        except Exception as e:
            error_msg = f"调用模型时出错: {str(e)}"
            logger.error(error_msg)
            return "抱歉，我遇到了一些技术问题，请稍后再试。"

        # 4. 添加助手回复到历史
        self.history.append({"role": "assistant", "content": reply})
        if self.debug:
            self._print_history()

        return reply

    def reset(self):
        """重置对话历史，仅保留系统提示词。"""
        self.history = [self.system_prompt]
        if self.debug:
            logger.info("对话历史已重置")

    def get_history(self) -> list:
        """返回当前对话历史的副本（只读）。"""
        return self.history.copy()


def main():
    """主函数：启动交互式命令行对话。"""
    personality = "你是一个友好的技术助教，擅长解释编程概念，回答时使用简洁清晰的中文。"
    bot = ChatBot(personality=personality, debug=True)

    print("机器人已启动，输入 'quit' 退出，输入 'reset' 重置对话。")
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "reset":
            bot.reset()
            print("对话已重置。")
            continue

        if not user_input:
            continue

        reply = bot.chat(user_input)
        print(f"机器人: {reply}")


if __name__ == "__main__":
    main()