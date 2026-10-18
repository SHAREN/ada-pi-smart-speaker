#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from pathlib import Path

from backend.audio_debug import AudioDebugSession


async def main() -> None:
    session = AudioDebugSession(Path('/home/renat/ada-pi'), 'smoke')
    await session.start()
    session.write_raw_mic(b'\x00\x00' * 16000)
    await asyncio.sleep(1)
    await session.close('smoke')
    print(session.dir)
    print(f'render_exists={session.render_path.exists()}')
    print(f'render_bytes={session.render_path.stat().st_size if session.render_path.exists() else 0}')
    print(f'mix_exists={session.mix_path.exists()}')


if __name__ == '__main__':
    asyncio.run(main())
