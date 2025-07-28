# controller.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from src.chat_module.intent_service import predict_intent
from src.chat_module.serializer import TextInput
import uvicorn

chat_router = APIRouter()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Mental Health Chatbot</title>
    </head>
    <body>
        <h1>Mental Health WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("wss://api-testing-ucy2.onrender.com/interact");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages');
                var message = document.createElement('li');
                var content = document.createTextNode(event.data);
                message.appendChild(content);
                messages.appendChild(message);
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText");
                wss.send(input.value);
                input.value = '';
                event.preventDefault();
            }
        </script>
    </body>
</html>
"""

@chat_router.get("/chat")
async def get():
    return HTMLResponse(html)


@chat_router.websocket("/interact")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("🧠 Hello! How can I help you today?")
    try:
        while True:
            data = await websocket.receive_text()
            input_text = TextInput(text=data)
            result = predict_intent(input_text)
            await websocket.send_json(result.dict())
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run("controller:chat_router", host="0.0.0.0", port=8000, reload=True)
