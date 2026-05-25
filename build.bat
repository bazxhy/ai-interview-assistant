@echo off
echo ========================================
echo   AI 面试助手 - 打包为可执行程序
echo ========================================
echo.

pyinstaller ^
    --name="AI面试助手" ^
    --onedir ^
    --console ^
    --clean ^
    --noconfirm ^
    --add-data=".env.example;." ^
    --hidden-import=websocket ^
    --hidden-import=sounddevice ^
    --hidden-import=numpy ^
    --hidden-import=openai ^
    --hidden-import=dotenv ^
    --hidden-import=faster_whisper ^
    --hidden-import=ctranslate2 ^
    --hidden-import=onnxruntime ^
    --hidden-import=keyboard ^
    --hidden-import=av ^
    --hidden-import=tokenizers ^
    --hidden-import=huggingface_hub ^
    --hidden-import=pyttsx3 ^
    --hidden-import=json ^
    --hidden-import=wave ^
    --hidden-import=threading ^
    --hidden-import=queue ^
    --exclude-module=matplotlib ^
    --exclude-module=pandas ^
    --exclude-module=PIL ^
    --exclude-module=cv2 ^
    --exclude-module=scipy ^
    main.py

echo.
echo ========================================
echo   打包完成!
echo   程序在: dist\AI面试助手\
echo   运行: dist\AI面试助手\AI面试助手.exe
echo   (请确保同级目录下有 .env 配置文件)
echo ========================================
pause
