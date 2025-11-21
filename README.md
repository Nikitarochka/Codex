# Codex

## PE5
- `pe5_direct.py` вызывает корреляцию PE5 через `PXFlow.dll`.
- Пример запуска: `python pe5_direct.py`.

## OLGAS
- `olgas_direct.py` вызывает OLGAS (`libolgas2000.dll`).
- DLL — 32‑битная Windows, поэтому нужен **Windows + 32‑битный Python** (совпадение разрядности с DLL).
- Путь к DLL можно задать переменной `OLGAS_DLL_PATH`, иначе берётся файл рядом со скриптом.
- Проверить разрядность Python: `python -c "import struct; print(struct.calcsize('P')*8)"`.
- Если разрядность не совпадает, установи Python нужной битности и запусти скрипт под Windows.
- Для отладки без DLL можно включить заглушку: `OLGAS_STUB=1 python olgas_direct.py` (подставит псевдорезультаты).
- Проверку платформы можно принудительно отключить (на свой риск) через `OLGAS_FORCE_LOAD=1`.
