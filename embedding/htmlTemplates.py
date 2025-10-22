css = """
<style>
.chat-message {
    padding: 0.6rem 0.8rem;
    border-radius: 0.6rem;
    margin-bottom: 0.7rem;
    line-height: 1.4;
    white-space: pre-wrap;
    font-size: 0.95rem;
}
.chat-message.user {
    background-color: #eef3ff;
    border: 1px solid #d9e2ff;
}
.chat-message.bot {
    background-color: #f7f7f9;
    border: 1px solid #e8e8ef;
}
</style>
"""

user_template = """
<div class="chat-message user">
    <strong>You</strong><br/>
    {{MSG}}
</div>
"""

bot_template = """
<div class="chat-message bot">
    <strong>Reviewer</strong><br/>
    {{MSG}}
</div>
"""
