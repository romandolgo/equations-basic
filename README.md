# Лабораторная работа №2. Разложения и решение СЛАУ

Реализации алгоритмов находятся в `src/`, эксперименты и отчёты по заданиям — в `notebooks/`.

| Часть | Файл |
|:-|:-|
| LUP-разложение | [`src/decomps/lup.py`](src/decomps/lup.py) |
| LDLT-разложение | [`src/decomps/ldlt.py`](src/decomps/ldlt.py) |
| QR-разложение | [`src/decomps/qr.py`](src/decomps/qr.py) |
| Треугольные системы | [`src/solvers/lower_solve.py`](src/solvers/lower_solve.py), [`src/solvers/upper_solve.py`](src/solvers/upper_solve.py) |
| Решение через разложения | [`src/solvers/lup_solve.py`](src/solvers/lup_solve.py), [`src/solvers/ldlt_solve.py`](src/solvers/ldlt_solve.py), [`src/solvers/qr_solve.py`](src/solvers/qr_solve.py) |
| Программа-решатель | [`src/solvers/slau_solver.py`](src/solvers/slau_solver.py) |
| Решатель МНК | [`src/mnk/lstsq.py`](src/mnk/lstsq.py) |
| Регуляризация Тихонова | [`src/mnk/tikhonov.py`](src/mnk/tikhonov.py) |
| Задание I. Обусловленность и устойчивость | [`notebooks/conditioning.ipynb`](notebooks/conditioning.ipynb) |
| Задание II. МНК и потеря точности | [`notebooks/lstsq_accuracy.ipynb`](notebooks/lstsq_accuracy.ipynb) |
| Задание III. Регуляризация и ошибки данных | [`notebooks/regularization.ipynb`](notebooks/regularization.ipynb) |
| Задание IV. Задача извне | [`notebooks/reservoirs.ipynb`](notebooks/reservoirs.ipynb) |

Модули импортируются от `src`: `from decomps.lup import lup_factor`.

Ноутбуки можно запускать из корня репозитория или из `notebooks/`: путь к `src/` добавляется в первой ячейке с кодом.

Публичные тесты: [`tests/test_public.py`](tests/test_public.py).

## Локальное тестирование

Из корня репозитория:

```shell
pip install -r requirements.txt
pytest
```

Тесты нужно запускать из корня репозитория. Отдельные тесты — по имени:
```shell
pytest -k lup
```

### Закрытые тесты

В CI дополнительно запускаются закрытые тесты. Они находятся в специальном Docker-образе и не входят в репозиторий с заданием.

Для локального запуска закрытого чекера требуется установленный Docker. Инструкции по установке доступны в [официальной документации](https://docs.docker.com/engine/install/).

Из корня репозитория выполните:

```shell
docker run --pull always --rm \
  -v "$(pwd)/src:/workspace/src:ro" \
  cr.ct.itmo.ru/numla/hw-checker/equations-basic:latest
```

Чекер содержит как публичные, так и закрытые тесты и запускает их вместе.

## Кодстайл

Проверяется `ruff` (неиспользуемые импорты и переменные):

```shell
ruff check src --select F401,F841
```
