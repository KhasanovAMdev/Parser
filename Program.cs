using CsvHelper;
using HtmlAgilityPack;
using MathNet.Numerics.Statistics;
using Microsoft.ML;
using Microsoft.ML.Data;
using ScottPlot;
using System.Globalization;
using System.Net;
using System.Text.RegularExpressions;

namespace RealEstateAnalysis
{
    public class ApartmentData
    {
        public string? Url { get; set; }
        public float Price { get; set; }
        public float Square { get; set; }
        public int Floor { get; set; }
        public int TotalFloors { get; set; }
        public string? District { get; set; }
        public int Rooms { get; set; }
    }

    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== Анализ рынка недвижимости Ульяновска ===");

            // 1. Парсинг данных
            Console.WriteLine("\n[1/6] Парсинг данных с Avito...");
            var apartments = await ParseAvito("ulyanovsk", pages: 3, useSavedHtml: true);

            // Фильтрация некорректных данных
            apartments = apartments.Where(a =>
                a.Price > 0 && a.Square > 0 && a.Rooms > 0 && a.Floor > 0
            ).ToList();

            Console.WriteLine($"Получено {apartments.Count} объявлений после фильтрации");

            // 2. Сохранение в CSV
            Console.WriteLine("\n[2/6] Сохранение данных в CSV...");
            SaveToCsv(apartments, "apartments.csv");

            // 3. Описательная статистика
            Console.WriteLine("\n[3/6] Расчет описательной статистики...");
            CalculateDescriptiveStats(apartments);

            // 4. Визуализация данных
            Console.WriteLine("\n[4/6] Визуализация данных...");
            GeneratePlots(apartments);

            // 5. Корреляционный анализ
            Console.WriteLine("\n[5/6] Корреляционный анализ...");
            CalculateCorrelations(apartments);

            // 6. Регрессионный анализ
            Console.WriteLine("\n[6/6] Построение регрессионной модели...");
            BuildRegressionModel(apartments);

            Console.WriteLine("\nАнализ завершен! Результаты сохранены в папке с программой.");
        }

        static async Task<List<ApartmentData>> ParseAvito(string city, int pages = 3, bool useSavedHtml = false)
        {
            var apartments = new List<ApartmentData>();
            var httpClient = new HttpClient();
            httpClient.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
            httpClient.DefaultRequestHeaders.Add("Accept-Language", "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7");

            for (int page = 1; page <= pages; page++)
            {
                Console.WriteLine($"Обработка страницы {page}...");
                string html;
                string debugFileName = $"debug_page_{page}.html";

                try
                {
                    if (useSavedHtml && File.Exists(debugFileName))
                    {
                        Console.WriteLine($"Используется сохраненный файл {debugFileName}");
                        html = await File.ReadAllTextAsync(debugFileName);
                    }
                    else
                    {
                        string url = $"https://www.avito.ru/{city}/kvartiry/prodam/vtorichka-ASgBAQICAUSSA8YQAUDmBxSMUg?p={page}";
                        html = await httpClient.GetStringAsync(url);

                        // Сохраняем HTML для отладки
                        await File.WriteAllTextAsync(debugFileName, html);
                        Console.WriteLine($"Страница сохранена в {debugFileName}");

                        // Добавляем случайную задержку
                        await Task.Delay(Random.Shared.Next(2000, 5000));
                    }

                    var doc = new HtmlDocument();
                    doc.LoadHtml(html);

                    // Проверяем, не блокирует ли Avito парсинг
                    if (doc.DocumentNode.InnerHtml.Contains("Доступ ограничен"))
                    {
                        Console.WriteLine("Обнаружена блокировка Avito. Попробуйте:");
                        Console.WriteLine("1. Использовать прокси");
                        Console.WriteLine("2. Увеличить задержки между запросами");
                        Console.WriteLine("3. Проверить User-Agent");
                        break;
                    }

                    var nodes = doc.DocumentNode.SelectNodes("//div[contains(@class, 'iva-item-root')]") ?? new HtmlNodeCollection(null);

                    if (nodes.Count == 0)
                    {
                        Console.WriteLine("Объявления не найдены. Возможные причины:");
                        Console.WriteLine("1. Изменилась структура сайта");
                        Console.WriteLine("2. Сработала защита от парсинга");
                        Console.WriteLine("3. Нет результатов по запросу");

                        // Сохраняем HTML для анализа
                        File.WriteAllText($"error_debug_page_{page}.html", html);
                        Console.WriteLine($"Ошибка сохранена в error_debug_page_{page}.html");
                        break;
                    }

                    foreach (var node in nodes)
                    {
                        try
                        {
                            // 1. Ссылка и общие параметры из title
                            var linkNode = node.SelectSingleNode(".//a[contains(@data-marker, 'item-title')]");
                            if (linkNode == null) continue;

                            var titleText = WebUtility.HtmlDecode(linkNode.GetAttributeValue("title", ""));
                            titleText = titleText.Replace("&nbsp;", " ");

                            // 2. Цена (несколько вариантов селекторов)
                            var priceNode = node.SelectSingleNode(".//meta[@itemprop='price']") ??
                                            node.SelectSingleNode(".//span[contains(@data-marker, 'price')]");

                            // 3. Площадь и комнаты (из title и отдельного блока)
                            var paramsNode = node.SelectSingleNode(".//div[contains(@class, 'living-params')]") ??
                                            node.SelectSingleNode(".//p[contains(@data-marker, 'item-specific-params')]");

                            // 4. Этаж (из title и отдельного блока)
                            var floorNode = node.SelectSingleNode(".//*[contains(text(), 'этаж')]") ??
                                           node.SelectSingleNode(".//span[contains(@data-marker, 'item-specific-params')]");

                            // 5. Район (новый селектор)
                            var districtNode = node.SelectSingleNode(".//div[contains(@class, 'geo-georeferences')]//span") ??
                                              node.SelectSingleNode(".//div[contains(@data-marker, 'item-address')]//span");

                            // Парсим данные
                            var apartment = new ApartmentData
                            {
                                Url = "https://www.avito.ru" + linkNode.GetAttributeValue("href", ""),
                                District = districtNode?.InnerText.Trim() ?? "Не указан"
                            };

                            // Обработка цены
                            if (priceNode != null)
                            {
                                string priceText = priceNode.GetAttributeValue("content", priceNode.InnerText);
                                apartment.Price = ParsePrice(priceText);
                            }

                            // Основной парсинг из title (если есть данные)
                            if (!string.IsNullOrEmpty(titleText))
                            {
                                // Парсинг комнат
                                var roomsMatch = Regex.Match(titleText, @"(\d+)-к");
                                if (roomsMatch.Success)
                                {
                                    apartment.Rooms = int.Parse(roomsMatch.Groups[1].Value);
                                }

                                // Парсинг площади
                                var squareMatch = Regex.Match(titleText, @"(\d+[.,]\d+|\d+)\s*м²");
                                if (squareMatch.Success)
                                {
                                    apartment.Square = float.Parse(squareMatch.Groups[1].Value.Replace(",", "."),
                                                      CultureInfo.InvariantCulture);
                                }

                                // Парсинг этажности
                                var floorMatch = Regex.Match(titleText, @"(\d+)/(\d+)\s*эт");
                                if (floorMatch.Success)
                                {
                                    apartment.Floor = int.Parse(floorMatch.Groups[1].Value);
                                    apartment.TotalFloors = int.Parse(floorMatch.Groups[2].Value);
                                }
                            }

                            // Дополнительный парсинг из отдельных блоков (если в title не нашли)
                            if (paramsNode != null && (apartment.Rooms == 0 || apartment.Square == 0))
                            {
                                var paramsText = paramsNode.InnerText;

                                if (apartment.Rooms == 0)
                                {
                                    var roomsMatch = Regex.Match(paramsText, @"(\d+)-к");
                                    if (roomsMatch.Success)
                                    {
                                        apartment.Rooms = int.Parse(roomsMatch.Groups[1].Value);
                                    }
                                }

                                if (apartment.Square == 0)
                                {
                                    var squareMatch = Regex.Match(paramsText, @"(\d+[.,]\d+|\d+)\s*м²");
                                    if (squareMatch.Success)
                                    {
                                        apartment.Square = float.Parse(squareMatch.Groups[1].Value.Replace(",", "."),
                                                          CultureInfo.InvariantCulture);
                                    }
                                }
                            }

                            // Дополнительный парсинг этажа (если не нашли в title)
                            if (floorNode != null && apartment.Floor == 0)
                            {
                                var floorMatch = Regex.Match(floorNode.InnerText, @"(\d+)\s*/\s*(\d+)");
                                if (floorMatch.Success)
                                {
                                    apartment.Floor = int.Parse(floorMatch.Groups[1].Value);
                                    apartment.TotalFloors = int.Parse(floorMatch.Groups[2].Value);
                                }
                            }
                            apartments.Add(apartment);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Ошибка при обработке объявления: {ex.Message}");
                            File.AppendAllText("parse_errors.log", $"[{DateTime.Now}] Страница {page}: {ex}\n");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка при загрузке страницы {page}: {ex.Message}");
                    File.AppendAllText("load_errors.log", $"[{DateTime.Now}] Страница {page}: {ex}\n");
                }
            }

            return apartments;
        }

        // Метод для парсинга цены с улучшенной обработкой
        private static float ParsePrice(string priceText)
        {
            if (string.IsNullOrWhiteSpace(priceText)) return 0;

            try
            {
                // Удаляем все нецифровые символы, кроме разделителей
                string cleanText = Regex.Replace(priceText, @"[^\d,.]", "");

                // Нормализуем разделитель десятичных
                cleanText = cleanText.Replace(",", ".");

                // Обработка миллионов/тысяч
                float multiplier = 1f;
                if (priceText.Contains("млн")) multiplier = 1000000f;
                else if (priceText.Contains("тыс")) multiplier = 1000f;

                if (float.TryParse(cleanText, NumberStyles.Any, CultureInfo.InvariantCulture, out float price))
                {
                    return price * multiplier;
                }
            }
            catch
            {
                Console.WriteLine($"Ошибка парсинга цены: {priceText}");
            }
            return 0;
        }

        static void SaveToCsv(List<ApartmentData> data, string filename)
        {
            try
            {
                using var writer = new StreamWriter(filename);
                using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);

                // Настройка заголовков
                csv.WriteHeader<ApartmentData>();
                csv.NextRecord();

                foreach (var item in data)
                {
                    csv.WriteRecord(item);
                    csv.NextRecord();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка сохранения CSV: {ex.Message}");
            }
        }

        static void CalculateDescriptiveStats(List<ApartmentData> apartments)
        {
            if (apartments == null || !apartments.Any())
            {
                Console.WriteLine("Нет данных для статистики");
                return;
            }

            var prices = apartments.Select(a => (double)a.Price).ToList();
            var squares = apartments.Select(a => (double)a.Square).ToList();
            var rooms = apartments.Select(a => (double)a.Rooms).ToList();
            var floors = apartments.Select(a => (double)a.Floor).ToList();

            Console.WriteLine("\n=== Описательная статистика ===");

            PrintStats("Цены (руб)", prices);
            PrintStats("Площадь (м²)", squares);
            PrintStats("Количество комнат", rooms);
            PrintStats("Этаж", floors);
        }

        static void PrintStats(string name, List<double> values)
        {
            Console.WriteLine($"\n{name}:");
            Console.WriteLine($"• Среднее: {values.Mean():0.##}");
            Console.WriteLine($"• Медиана: {values.Median():0.##}");
            Console.WriteLine($"• Стандартное отклонение: {values.StandardDeviation():0.##}");
            Console.WriteLine($"• Минимум: {values.Min():0.##}");
            Console.WriteLine($"• Максимум: {values.Max():0.##}");
            Console.WriteLine($"• Квартили (25%/50%/75%): {values.Quantile(0.25):0.##} / {values.Quantile(0.5):0.##} / {values.Quantile(0.75):0.##}");
        }

        static void GeneratePlots(List<ApartmentData> apartments)
        {
            try
            {
                // Гистограмма цен
                var prices = apartments.Select(a => (double)a.Price / 1000000).ToArray();
                var plt1 = new Plot();
                plt1.Title("Распределение цен на квартиры");
                plt1.XLabel("Цена (млн руб)");
                plt1.YLabel("Количество");
                var hist = ScottPlot.Statistics.Histogram.WithBinCount(10, prices);
                var barPlot = plt1.Add.Bars(hist.Bins, hist.Counts);
                foreach (var bar in barPlot.Bars)
                {
                    bar.Size = hist.FirstBinSize * .8;
                }
                plt1.Axes.Margins(bottom: 0);
                plt1.SavePng("price_distribution.png", 800, 600);

                // Зависимость цены от площади
                var plt2 = new Plot();
                plt2.Title("Зависимость цены от площади");
                plt2.XLabel("Площадь (м²)");
                plt2.YLabel("Цена (млн руб)");
                var sp = plt2.Add.ScatterPoints(apartments.Select(a => (double)a.Square).ToArray(), prices);
                sp.MarkerSize = 10;
                plt2.SavePng("price_vs_area.png",800,600);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка генерации графиков: {ex.Message}");
            }
        }

        static void CalculateCorrelations(List<ApartmentData> apartments)
        {
            var data = new Dictionary<string, double[]>
            {
                { "Цена", apartments.Select(a => (double)a.Price).ToArray() },
                { "Площадь", apartments.Select(a => (double)a.Square).ToArray() },
                { "Комнаты", apartments.Select(a => (double)a.Rooms).ToArray() },
                { "Этаж", apartments.Select(a => (double)a.Floor).ToArray() }
            };

            var corrMatrix = Correlation.PearsonMatrix(data.Values.ToArray());

            Console.WriteLine("\n=== Матрица корреляций ===");
            Console.WriteLine("               Цена   Площадь  Комнаты  Этаж");
            for (int i = 0; i < data.Count; i++)
            {
                Console.Write($"{data.Keys.ElementAt(i),-15}");
                for (int j = 0; j < data.Count; j++)
                {
                    Console.Write($"{corrMatrix[i, j]:0.00}    ");
                }
                Console.WriteLine();
            }
        }

        static void BuildRegressionModel(List<ApartmentData> apartments)
        {
            try
            {
                var mlContext = new MLContext(seed: 42);

                var trainingData = apartments.Select(a => new ApartmentTrainingData
                {
                    Label = a.Price,
                    Square = a.Square,
                    Rooms = (float)a.Rooms,
                    Floor = (float)a.Floor,
                    District = a.District ?? "Не указан"
                }).ToList();

    
                var data = mlContext.Data.LoadFromEnumerable(trainingData);


                var pipeline = mlContext.Transforms
                    .Categorical.OneHotEncoding("DistrictEncoded", "District")
                    .Append(mlContext.Transforms.Concatenate(
                        "Features",
                        nameof(ApartmentTrainingData.Square),
                        nameof(ApartmentTrainingData.Rooms),
                        nameof(ApartmentTrainingData.Floor),
                        "DistrictEncoded")
                    )
                    .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(
                        labelColumnName: "Label",
                        featureColumnName: "Features"));

   
                var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
   
                var model = pipeline.Fit(trainTestSplit.TrainSet);

                var predictions = model.Transform(trainTestSplit.TestSet);
                var metrics = mlContext.Regression.Evaluate(predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score");

                Console.WriteLine("\n=== Регрессионная модель ===");
                Console.WriteLine($"R²: {metrics.RSquared:0.###}");
                Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.###} руб.");
                Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:0.###} руб.");

                var predictionEngine = mlContext.Model.CreatePredictionEngine<ApartmentTrainingData, PricePrediction>(model);

                var sample = trainingData[Random.Shared.Next(trainingData.Count)];
                var prediction = predictionEngine.Predict(sample);

                Console.WriteLine("\nПример прогноза:");
                Console.WriteLine($"• Реальная цена: {sample.Label:N0} руб.");
                Console.WriteLine($"• Прогнозируемая: {prediction.PredictedPrice:N0} руб.");
                Console.WriteLine($"• Параметры: {sample.Square} м², {sample.Rooms}к, этаж {sample.Floor}, район: {sample.District}");

                mlContext.Model.Save(model, data.Schema, "real_estate_model.zip");
                Console.WriteLine("\nМодель сохранена в файл: real_estate_model.zip");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка построения модели: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Внутренняя ошибка: {ex.InnerException.Message}");
                }
            }
        }

        public class ApartmentTrainingData
        {
            [LoadColumn(0)]
            public float Label { get; set; } 

            [LoadColumn(1)]
            public float Square { get; set; }

            [LoadColumn(2)]
            public float Rooms { get; set; }

            [LoadColumn(3)]
            public float Floor { get; set; }

            [LoadColumn(4)]
            public string? District { get; set; }
        }

        public class PricePrediction
        {
            [ColumnName("Score")]
            public float PredictedPrice { get; set; }
        }
    }
}