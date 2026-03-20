import os
import time
import asyncio
import threading
from collections import deque
from dotenv import load_dotenv

load_dotenv()


class RateLimiter:
    """基于滑动窗口日志的线程安全 RPM 限速器。超速时排队等待，不报错。"""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.window = 60.0
        self.lock = threading.Lock()
        self.timestamps = deque()

    def acquire(self):
        """同步阻塞直到获得一个槽位。供工作线程调用。"""
        while True:
            with self.lock:
                now = time.time()
                while self.timestamps and self.timestamps[0] <= now - self.window:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return
                wait_time = self.timestamps[0] - (now - self.window)
            time.sleep(max(wait_time, 0.1))

    async def async_acquire(self):
        """异步等待直到获得一个槽位。供 FastAPI 异步端点调用，不阻塞事件循环。"""
        while True:
            with self.lock:
                now = time.time()
                while self.timestamps and self.timestamps[0] <= now - self.window:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return
                wait_time = self.timestamps[0] - (now - self.window)
            await asyncio.sleep(max(wait_time, 0.1))


# 从 .env 读取 RPM 配置，0 或未设置表示不限速
_pro_rpm = int(os.getenv("MODEL_PRO_RPM", "0"))
_flash_rpm = int(os.getenv("MODEL_FLASH_RPM", "0"))
_instance_rpm = int(os.getenv("INSTANCE_RPM", "0"))

pro_limiter = RateLimiter(_pro_rpm) if _pro_rpm > 0 else None
flash_limiter = RateLimiter(_flash_rpm) if _flash_rpm > 0 else None
instance_limiter = RateLimiter(_instance_rpm) if _instance_rpm > 0 else None
