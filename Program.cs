using CsvHelper;
using HtmlAgilityPack;
using MathNet.Numerics.Statistics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
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

            Console.WriteLine("\n[1/6] Парсинг данных с Avito...");
            var apartments = await ParseAvito("ulyanovsk", pages: 5, useSavedHtml: true);

            apartments = apartments.Where(a =>
                a.Price > 0 && a.Square > 0 && a.Rooms > 0 && a.Floor > 0
            ).ToList();

            Console.WriteLine($"Получено {apartments.Count} объявлений после фильтрации");

            Console.WriteLine("\n[2/6] Сохранение данных в CSV...");
            SaveToCsv(apartments, "apartments.csv");

            Console.WriteLine("\n[3/6] Расчет описательной статистики...");
            CalculateDescriptiveStats(apartments);

            Console.WriteLine("\n[4/6] Визуализация данных...");
            GeneratePlots(apartments);

            Console.WriteLine("\n[5/6] Корреляционный анализ...");
            CalculateCorrelations(apartments);

            Console.WriteLine("\n[6/6] Построение прогнозирующих моделей...");
            BuildModels(apartments);

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

                        await File.WriteAllTextAsync(debugFileName, html);
                        Console.WriteLine($"Страница сохранена в {debugFileName}");

                        await Task.Delay(Random.Shared.Next(2000, 5000));
                    }

                    var doc = new HtmlDocument();
                    doc.LoadHtml(html);

                    if (doc.DocumentNode.InnerHtml.Contains("Доступ ограничен"))
                    {
                        break;
                    }

                    var nodes = doc.DocumentNode.SelectNodes("//div[contains(@class, 'iva-item-root')]") ?? new HtmlNodeCollection(null);

                    if (nodes.Count == 0)
                    {
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
                                    var squareMatch = Regex.Match(paramsText, @"(\d+[.,]\d+|\d+)\s*м^2");
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
            PrintStats("Площадь (м^2)", squares);
            PrintStats("Количество комнат", rooms);
            PrintStats("Этаж", floors);
        }

        static void PrintStats(string name, List<double> values)
        {
            Console.WriteLine($"\n{name}:");
            Console.WriteLine($"Среднее: {values.Mean():0.##}");
            Console.WriteLine($"Медиана: {values.Median():0.##}");
            Console.WriteLine($"Стандартное отклонение: {values.StandardDeviation():0.##}");
            Console.WriteLine($"Минимум: {values.Min():0.##}");
            Console.WriteLine($"Максимум: {values.Max():0.##}");
            Console.WriteLine($"Квартили (25%/50%/75%): {values.Quantile(0.25):0.##} / {values.Quantile(0.5):0.##} / {values.Quantile(0.75):0.##}");
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
                plt2.XLabel("Площадь (м^2)");
                plt2.YLabel("Цена (млн руб)");
                var sp = plt2.Add.ScatterPoints(apartments.Select(a => (double)a.Square).ToArray(), prices);
                sp.MarkerSize = 10;
                plt2.SavePng("price_vs_area.png", 800, 600);
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

        static void BuildModels(List<ApartmentData> apartments)
        {
            try
            {
                // Функция предобработки районов
                string GetDistrictGroup(string fullAddress)
                {
                    if (string.IsNullOrEmpty(fullAddress)) return "Не указан";
                    var address = fullAddress.Split(',')[0].Trim();
                    return address switch
                    {
                        string s when s.StartsWith("мкр-н") => s.Replace("мкр-н", "").Trim(),
                        string s when s.StartsWith("пр-т") => s.Replace("пр-т", "").Trim(),
                        string s when s.StartsWith("ул.") => s.Replace("ул.", "").Trim(),
                        _ => address
                    };
                }

                var mlContext = new MLContext(seed: 42);

                // Подготовка данных
                var trainingData = apartments.Select(a => new ApartmentTrainingData
                {
                    Label = a.Price,
                    Square = a.Square,
                    Rooms = (float)a.Rooms,
                    Floor = (float)a.Floor,
                    District = GetDistrictGroup(a.District)
                }).ToList();

                var data = mlContext.Data.LoadFromEnumerable(trainingData);
                var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

                // Список моделей для сравнения
                var models = new List<(string Name, IEstimator<ITransformer> Pipeline)>
        {
            ("Poisson Regression", mlContext.Regression.Trainers.LbfgsPoissonRegression()),
            ("Fast Tree", mlContext.Regression.Trainers.FastTree()),
            ("Fast Forest", mlContext.Regression.Trainers.FastForest()),
            ("Online Gradient Descent", mlContext.Regression.Trainers.OnlineGradientDescent()),
            ("LightGBM", mlContext.Regression.Trainers.LightGbm())
        };

                var results = new List<ModelResult>();

                // Обучение и оценка всех моделей
                foreach (var model in models)
                {
                    var pipeline = mlContext.Transforms
                        .Categorical.OneHotEncoding("DistrictEncoded", "District")
                        .Append(mlContext.Transforms.NormalizeMinMax(
                            outputColumnName: "NormalizedSquare",
                            inputColumnName: nameof(ApartmentTrainingData.Square)))
                        .Append(mlContext.Transforms.NormalizeMinMax(
                            outputColumnName: "NormalizedRooms",
                            inputColumnName: nameof(ApartmentTrainingData.Rooms)))
                        .Append(mlContext.Transforms.NormalizeMinMax(
                            outputColumnName: "NormalizedFloor",
                            inputColumnName: nameof(ApartmentTrainingData.Floor)))
                        .Append(mlContext.Transforms.Concatenate(
                            "Features",
                            "NormalizedSquare",
                            "NormalizedRooms",
                            "NormalizedFloor",
                            "DistrictEncoded"))
                        .Append(model.Pipeline);

                    Console.WriteLine($"\nОбучение модели: {model.Name}...");

                    var trainedModel = pipeline.Fit(trainTestSplit.TrainSet);
                    var predictions = trainedModel.Transform(trainTestSplit.TestSet);
                    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

                    results.Add(new ModelResult(
                        model.Name,
                        trainedModel,
                        metrics.RSquared,
                        metrics.RootMeanSquaredError,
                        metrics.MeanAbsoluteError
                    ));

                    Console.WriteLine($"Метрики {model.Name}:");
                    Console.WriteLine($"R^2: {metrics.RSquared:0.000}, RMSE: {metrics.RootMeanSquaredError:0.0}, MAE: {metrics.MeanAbsoluteError:0.0}");
                }

                // Выбор лучшей модели по R^2
                var bestModel = results.OrderByDescending(r => r.RSquared).First();

                Console.WriteLine("\n=== Лучшая модель ===");
                Console.WriteLine($"Название: {bestModel.Name}");
                Console.WriteLine($"R^2: {bestModel.RSquared:0.000}");
                Console.WriteLine($"RMSE: {bestModel.RMSE:0.0} руб.");
                Console.WriteLine($"MAE: {bestModel.MAE:0.0} руб.");

                // Сохранение лучшей модели
                mlContext.Model.Save(bestModel.Model, data.Schema, "best_real_estate_model.zip");
                Console.WriteLine("\nЛучшая модель сохранена в файл: best_real_estate_model.zip");

                // Пример прогноза лучшей модели
                var predictionEngine = mlContext.Model.CreatePredictionEngine<ApartmentTrainingData, PricePrediction>(bestModel.Model);
                var sample = trainingData[Random.Shared.Next(trainingData.Count)];
                var prediction = predictionEngine.Predict(sample); 

                Console.WriteLine("\nПример прогноза лучшей модели:");
                Console.WriteLine($"Реальная цена: {sample.Label:N0} руб.");
                Console.WriteLine($"Прогнозируемая: {prediction.PredictedPrice:N0} руб.");
                Console.WriteLine($"Параметры: {sample.Square} м^2, {sample.Rooms}к, этаж {sample.Floor}, район: {sample.District}");

                // Вывод сравнения всех моделей
                Console.WriteLine("\nСравнение моделей:");
                Console.WriteLine(" Модель / R^2 / RMSE / MAE ");
                foreach (var result in results.OrderByDescending(r => r.RSquared))
                {
                    Console.WriteLine($" {result.Name,-26} / {result.RSquared:0.000} / {result.RMSE:9.0} руб. / {result.MAE:9.0} руб. ");
                }
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

        // Класс для хранения результатов моделей
        public class ModelResult
        {
            public string Name { get; }
            public ITransformer Model { get; }
            public double RSquared { get; }
            public double RMSE { get; }
            public double MAE { get; }

            public ModelResult(string name, ITransformer model, double rSquared, double rmse, double mae)
            {
                Name = name;
                Model = model;
                RSquared = rSquared;
                RMSE = rmse;
                MAE = mae;
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