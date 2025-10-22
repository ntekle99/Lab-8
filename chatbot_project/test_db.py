import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = "mysql+asyncmy://chatuser:chatpass@localhost:3306/groupchat"

async def test_connection():
    engine = create_async_engine(DATABASE_URL, echo=True)
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 'Connected to MySQL via asyncmy!'"))
        print(result.scalar())
    await engine.dispose()

asyncio.run(test_connection())
