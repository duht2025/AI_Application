"""
QARobot_QwenBased - 基于 Qwen 模型（通过 Ollama 本地服务）的对话机器人
提供带历史记忆、人设设定、历史裁剪和异常处理的聊天功能。
"""

import json
import logging
from openai import OpenAI

# 可选：配置日志（便于生产环境调试）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)    # 运行时用代码名替换  


class ChatBot:
    """
    一个基于 OpenAI 兼容 API（如 Ollama）的聊天机器人。
    维护对话历史，支持人设设定和历史消息裁剪。
    """

    def __init__(self, personality: str, model: str = "qwen2.5:7b", max_history_len: int = 10,
                 temperature: float = 0.7, debug: bool = True):
        """
        初始化聊天机器人。

        :param personality: 人设描述（系统提示词）
        :param model: 使用的模型名称（需与 Ollama 中的模型名一致）
        :param max_history_len: 保留的最近对话轮数（不包括系统提示词）
        :param temperature: 生成回复的随机性（0~1之间）
        :param debug: 是否打印详细调试信息（包括历史内容）
        """
        self.system_prompt = {"role": "system", "content": personality}
        self.history = [self.system_prompt]  # 对话历史，始终以 system prompt 开头
        self.model = model
        self.max_history_len = max_history_len
        self.temperature = temperature
        self.debug = debug

        # 初始化 OpenAI 客户端，指向本地 Ollama 服务
        try:
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # Ollama 默认不需要有效 key，但必须提供
                timeout=120.0       # 设置请求超时时间
            )
            if self.debug:
                logger.info("Ollama 客户端初始化成功，base_url: http://localhost:11434/v1")
        except Exception as e:
            logger.error(f"初始化 OpenAI 客户端失败: {e}")
            raise  # 若客户端无法初始化，则向上抛出异常，阻止继续运行

    def _trim_history(self):
        """
        裁剪对话历史，保留系统提示词和最近的 max_history_len 条消息。
        此方法会在每次用户输入后调用，确保历史长度不超过限制。
        """
        # 总消息数（包括 system prompt）应 <= max_history_len + 1
        if len(self.history) > self.max_history_len + 1:
            # 保留 system prompt + 最近 max_history_len 条消息
            self.history = [self.system_prompt] + self.history[-(self.max_history_len):]
            if self.debug:
                logger.debug("历史消息已裁剪，当前长度: %d", len(self.history))

    def _print_history(self):
        """打印当前对话历史（调试用），每条消息只显示前50个字符。"""
        if not self.debug:
            return
        print("\n--- 当前对话历史 ---")
        for i, msg in enumerate(self.history):
            role = msg["role"]
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"{i+1}. {role}: {content}")
        print("=" * 50)

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
                stream=False
            )
            reply = response.choices[0].message.content
        except Exception as e:
            # 捕获所有可能的异常（网络问题、API错误、超时等）
            error_msg = f"调用模型时出错: {str(e)}"
            logger.error(error_msg)
            # 返回友好的错误信息给用户
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
    # 创建一个具有“友好助教”人设的机器人
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