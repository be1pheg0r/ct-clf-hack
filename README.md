# ConvSensus - автоматический скрининг КТ ОГК

ConvSensus - система автоматического анализа компьютерной томографии органов грудной клетки, использующая SoTA CNN модели для выявления патологий, включая COVID-19, пневмонию и т.д.
![ConvSensus](https://sun9-35.userapi.com/s/v1/if2/Wq3Xaem_G3pbtF7gSE7-etYLsFWnl4L7E-Y6K7HEo2rwMLC8b4CN19sNhdJ5ko3cJAQ-UBSgO2nZ_bq_tDLy9GV4.jpg?quality=95&as=32x15,48x22,72x34,108x50,160x75,240x112,360x168,480x224,540x252,640x299,720x337,813x380&from=bu&cs=813x0)

В проекте используются архитектуры моделей: 
- ResNet50
- DenseNet121
- InceptionV3
- ConVNext

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/be1pheg0r/ct-clf-hack.git
    cd ct-clf-hack
    ```

2. Запустите сборку и запуск контейнеров:
   ```bash
   make docker-up-fullstack
   ```
   
3. Для Web перейдите по адресу по свободному порту (вывод в консоли)

4. Запроса через api используйте http://localhost:{PORT}/process-image

5. Для остановки контейнеров:
   ```bash
   make docker-down
   ```
   
## Использование
1. Загрузите архив с DICOM файлами или 3D .nii файлом через веб-интерфейс или API.
2. Дождитесь завершения анализа.
3. Просмотрите результаты на веб-интерфейсе или получите их через API.