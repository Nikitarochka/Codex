# Просмотр и выводы lab8_eurosat

## Как посмотреть блокнот в текстовом виде
- Установите зависимости: `python3 -m pip install nbformat nbconvert`.
- Сконвертируйте ноутбук в Markdown одной командой: `jupyter nbconvert --to markdown lab8_eurosat.ipynb --stdout`.
- Если nbconvert недоступен, можно открыть выводы прямо из Python:
  ```python
  import nbformat
  nb = nbformat.read('lab8_eurosat.ipynb', as_version=4)
  for cell in nb.cells[-4:]:
      if cell.get('source'):
          print('---')
          print(''.join(cell['source']))
  ```

## Ключевые числа текущего прогона
- 3 эпохи на подвыборке 600 снимков: потери на обучении упали с ~2.00 до 1.04, точность выросла до 0.681; на проверке потеря снизилась до 0.8860 при точности 0.7889.
- Проверка на тесте: `poterja_test = 0.8196`, `tochnost_test = 0.7222`.
- Итоговые выводы в ноутбуке: модель усвоила классы на подвыборке, но для точности 0.85+ на тесте пригодятся больше эпох и/или частичная разморозка бэкбона.
