# Как выбрать тип?
### Для начала создайте папку `build`:
`mkdir build`
### Сборка с double (по умолчанию):
```
cmake -B build
make -C build
```
### Сборка с double:
```
cmake -B build -DTYPE=double
make -C build
```
### Сборка с float:
```
cmake -B build -DTYPE=float
make -C build
```
### Запуск программы:
`./build/task1`
### Удаление бинарника и временных файлов:
`rm -rf build`
## Результаты:
### double:
`1.56646e-10`
### float:
`-0.0277862`