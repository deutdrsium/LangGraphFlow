from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskData(BaseModel):
    task_id: str
    question_content: str
    answer: str

@app.post("/api/task_data")
async def receive_task_data(data: TaskData):
    print(f"\n--- New Data Received! ---")
    print(f"Task ID: {data.task_id}")
    print(f"Question Length: {len(data.question_content)} chars")
    print(f"Question Preview: {data.question_content[:150]}...")
    print(f"Answer Preview: {data.answer[:150]}...")
    print("--------------------------\n")
    return {"status": "success", "message": "Data received successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
