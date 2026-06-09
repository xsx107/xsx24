"""
Ollama API测试脚本（任务1）
AI辅助生成：接口调用与异常处理部分
"""
from langchain_community.llms import Ollama

# 初始化本地大模型
llm = Ollama(model="deepseek-r1:7b")

# 测试提问
try:
    response = llm.invoke("你好，请简单介绍一下RAG技术")
    print("✅ Ollama调用成功！")
    print("回答：", response)
except Exception as e:
    print("❌ Ollama调用失败：", e)