<!-- README.md -->

<h1 align="center">📈 Joke Concept Graphs</h1>

<p align="center">
  <em>Набор инструментов для обработки русскоязычных текстов (шуточных и литературных), извлечения концептов и построения графов отношений между ними.</em>
</p>

<hr/>

<h2>📂 Структура репозитория</h2>
<ul>
  <li><code>collected_graphs/</code> — сохранённые графы концептов (<code>.gpickle</code>)</li>
  <li><code>features/</code> — файлы с вычисленными признаками (<code>.csv</code>)</li>
  <li><code>graph/</code> — модули для работы с графами</li>
  <li><code>input_files/</code> — исходные тексты (<code>jokes.txt</code>, <code>lit_samples.txt</code> и др.)</li>
  <li><code>mark/</code> — энтропия и паттерны текстов</li>
  <li><code>output_files/</code> — результаты предобработки и концепты</li>
  <li><code>build_save_graphs.py</code> — скрипт: от предобработки до сохранения графов</li>
  <li><code>features.py</code> — вычисление признаков из текстов/графов</li>
  <li><code>features_utils.py</code> — вспомогательные функции для признаков</li>
  <li><code>imports.py</code> — импорты библиотек и ключевые параметры</li>
  <li><code>joke_concept.py</code> — извлечение и нормализация концептов из шуток</li>
  <li><code>lit_concept.py</code> — извлечение и нормализация концептов из литературы</li>
  <li><code>mark.py</code> — скрипт простенькой цепи Маркова</li>
  <li><code>relations.py</code> — извлечение отношений (триплетов)</li>
  <li><code>requirements.txt</code> — Python-зависимости</li>
</ul>

<hr/>

<h2>📊 Анализ</h2>
<p>
  Построенные графы (<code>collected_graphs/</code>).
</p>
<p>
  Выделенные признаки (<code>features/</code>).
</p>

<hr/>

<h2>📖 Форматы данных</h2>
<table>
  <thead>
    <tr>
      <th>Тип данных</th>
      <th>Расширение</th>
      <th>Описание</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Вход</td>
      <td><code>.txt</code></td>
      <td>Один документ на строку</td>
    </tr>
    <tr>
      <td>Признаки</td>
      <td><code>.csv</code></td>
      <td>Табличные данные со столбцами-метриками</td>
    </tr>
    <tr>
      <td>Графы</td>
      <td><code>.gpickle</code></td>
      <td>Узлы (концепты) и рёбра (отношения)</td>
    </tr>
  </tbody>
</table>

<hr/>
