using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;

// See "A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm",
// M. Riedmiller and H. Braun,
// Proceedings of the 1993 IEEE International Conference on Neural Networks,
// pp. 586-591
// This is the orginal version of the algorithm. There are many later variations.

namespace ResilientBackProp
{
    internal class RpropProgram
    {
        private static double[][] trainData;
        private static double[][] testData;

        const int numInput = 7959; // number features
        const int numOutput = 64; // number of classes for Y
        const int numHidden = numOutput * 2;

        public static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network with Resilient Back-Propagation (RPROP) training demo");

            LoadData();

            Console.WriteLine("Create neural network");
            int[] sizes = {numInput, numHidden, numOutput};
            NeuralNetwork nn = new NeuralNetwork(sizes);
            nn.Save("before_test.dat");

            const int maxEpochs = 1000;
            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);

            Console.WriteLine("\nStarting RPROP training");
            nn.multiThread = true;
            double[] weights = nn.TrainRPROP(trainData, maxEpochs, testData); // RPROP
            nn.Save("after_test.dat");
            Console.WriteLine("Done");
            Console.WriteLine("\nFinal neural network model weights:\n");
            // ShowVector(weights, 4, 10, true);

            double trainAcc = nn.Accuracy(trainData, weights);
            Console.WriteLine("\nAccuracy on training data = " +
                              trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData, weights);
            Console.WriteLine("\nAccuracy on test data = " +
                              testAcc.ToString("F4"));

            Console.WriteLine("\nEnd neural network with Resilient Propagation demo\n");
            Console.ReadLine();
        } // Main

        public static void ShowData(double[][] data, int numRows,
            int decimals, bool indices)
        {
            int len = data.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < data[i].Length; ++j)
                {
                    double v = data[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "    ");
                }
                Console.WriteLine("");
            }
            Console.WriteLine(". . .");
            int lastRow = data.Length - 1;
            if (indices)
                Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
            for (int j = 0; j < data[lastRow].Length; ++j)
            {
                double v = data[lastRow][j];
                if (v >= 0.0)
                    Console.Write(" "); // '+'
                Console.Write(v.ToString("F" + decimals) + "    ");
            }
            Console.WriteLine("\n");
        }

        public static void ShowVector(double[] vector, int decimals,
            int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }
            if (newLine)
                Console.WriteLine("");
        }

        protected static void LoadData()
        {
            DataContractJsonSerializer jsonFormatter =
                new DataContractJsonSerializer(typeof(TemporaryJsonClassDA[]));
            DataContractJsonSerializer jsonFormatterUID =
                new DataContractJsonSerializer(typeof(TemporaryJsonClassI[]));

            const int learn_file_count = 2;//315;
            const int test_file_count = 1;//90;

            List<double[]> learn_data = new List<double[]>();
            TemporaryJsonClassDA[] inner;
            TemporaryJsonClassI[] outer;

            double[] datum = new double[numInput + numOutput];
            for (int i = 0; i < learn_file_count; i++)
            {
                Console.WriteLine("Load Learn {0}", i);
                string s = "D:/_dev_stand/fp_txt_to_dat/output_json/learn-" + i + ".json";
                using (FileStream fs = new FileStream(s, FileMode.Open))
                {
                    inner = (TemporaryJsonClassDA[]) jsonFormatter.ReadObject(fs);
                }

                s = "D:/_dev_stand/fp_txt_to_dat/output_json/learn-" + i + ".out.json";
                using (FileStream fs = new FileStream(s, FileMode.Open))
                {
                    outer = (TemporaryJsonClassI[]) jsonFormatterUID.ReadObject(fs);
                }
                if (inner.Length != outer.Length)
                {
                    throw new Exception();
                }

                for (int j = 0; j < inner.Length; j++)
                {
                    if (inner[j].v.Length != numInput)
                    {
                        throw new Exception();
                    }
                    Array.Copy(inner[j].v, datum, numInput);
                    for (int n = numInput; n < numInput + numOutput; n++)
                    {
                        datum[n] = 0;
                    }
                    if (outer[j].v >= numOutput)
                    {
                        throw new Exception();
                    }
                    datum[outer[j].v + numInput] = 1;
                    learn_data.Add(datum);
                }
            }
            trainData = learn_data.ToArray();
            learn_data.Clear();

            for (int i = 0; i < test_file_count; i++)
            {
                Console.WriteLine("Load Test1 {0}", i);
                string s = "D:/_dev_stand/fp_txt_to_dat/output_json/test1-" + i + ".json";
                using (FileStream fs = new FileStream(s, FileMode.Open))
                {
                    inner = (TemporaryJsonClassDA[]) jsonFormatter.ReadObject(fs);
                }

                s = "D:/_dev_stand/fp_txt_to_dat/output_json/test1-" + i + ".out.json";
                using (FileStream fs = new FileStream(s, FileMode.Open))
                {
                    outer = (TemporaryJsonClassI[]) jsonFormatterUID.ReadObject(fs);
                }
                if (inner.Length != outer.Length)
                {
                    throw new Exception();
                }

                for (int j = 0; j < inner.Length; j++)
                {
                    if (inner[j].v.Length != numInput)
                    {
                        throw new Exception();
                    }
                    Array.Copy(inner[j].v, datum, numInput);
                    for (int n = numInput; n < numInput + numOutput; n++)
                    {
                        datum[n] = 0;
                    }
                    if (outer[j].v >= numOutput)
                    {
                        throw new Exception();
                    }
                    datum[outer[j].v + numInput] = 1;
                    learn_data.Add(datum);
                }
            }
            testData = learn_data.ToArray();
        }
    } // Program

    [DataContract]
    internal class TemporaryJsonClassDA
    {
        [DataMember] public double[] v;
    }

    [DataContract]
    internal class TemporaryJsonClassI
    {
        [DataMember] public int v;
    }

    public struct WeightComposite
    {
        public double[][] Weights;
        public double[] Biases;
    }

    public struct ThreadInputDatum
    {
        public double[][] trainDatum;
        public WeightComposite[] allGradsAcc;
        public double[][] field;

        public double[] xValues;
        public double[] tValues;

        public double delim1;
        public double delim2;
        public double[] sumSquaredErrors;
    }

    public struct RMSEThreadInputDatum
    {
        public double[][] trainDatum;
        public double[] xValues;
        public double[] tValues;
        public double[][] field;
        public double delim1;
        public double delim2;
        public double[] sumSquaredErrors;
    }

    public class NeuralNetwork : AbstractNeuralNetwork
    {
        public NeuralNetwork(IReadOnlyList<int> sizes) : base(sizes)
        {
        }

        protected override double[] ActivateFunction(IReadOnlyList<double>  rawValues, int layerId)
        {
            if (layerId >= this.LayerCount - 1)
            {
                return NeuralNetwork.Softmax(rawValues);
            }

            double[] values = new double[rawValues.Count];
            for (int i = 0; i < rawValues.Count; i++)
            {
                values[i] = NeuralNetwork.HyperTan(rawValues[i]);
            }

            return values;
        }

        protected override double[][] CalculateGradTerms(double[][] rawValues,
            IReadOnlyList<double> tValues)
        {
            double[][] gradTerms = new double[rawValues.Length][];
            for (int layerId = this.LayerCount - 1; layerId > 0; layerId--)
            {
                gradTerms[layerId] = layerId < this.LayerCount - 1
                    ? CalculateGradTermsForNonLast(rawValues[layerId], this.Neurons[layerId + 1].Weights,
                        gradTerms[layerId + 1])
                    : CalculateGradTermsForLast(rawValues[layerId], tValues);
            }

            return gradTerms;
        }

        private static double[] CalculateGradTermsForLast(IReadOnlyList<double> rawValues,
            IReadOnlyList<double> tValues)
        {
            double[] gradTerms = new double[rawValues.Count];
            for (int i = 0; i < rawValues.Count; ++i)
            {
                // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                double derivative = (1 - rawValues[i]) * rawValues[i];
                // careful with O-T vs. T-O, O-T is the most usual
                gradTerms[i] = derivative * (rawValues[i] - tValues[i]);
            }

            return gradTerms;
        }

        private static double[] CalculateGradTermsForNonLast(IReadOnlyList<double> rawValues,
            IReadOnlyList<double[]> nextNeuronLayerWeights,
            IReadOnlyList<double> nextGradTerms)
        {
            double[] gradTerms = new double[rawValues.Count];
            for (int i = 0; i < rawValues.Count; ++i)
            {
                double value = rawValues[i];
                // derivative of tanh = (1 - y) * (1 + y)
                double derivative = (1 - value) * (1 + value);
                double sum = nextGradTerms.Select((t, j) => t * nextNeuronLayerWeights[j][i]).Sum();
                // each hidden delta is the sum of this.sizes[2] terms
                gradTerms[i] = derivative * sum;
            }

            return gradTerms;
        }

        protected static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            return x > 20.0 ? 1.0 : Math.Tanh(x);
        }

        protected static double[] Softmax(IReadOnlyList<double> oSums)
        {
            // does all output nodes at once so scale doesn't have to be re-computed each time
            // determine max output-sum
            double max = oSums[0];
            max = oSums.Concat(new[] {max}).Max();

            // determine scaling factor -- sum of exp(each val - max)
            double scale = oSums.Sum(t => Math.Exp(t - max));

            double[] result = new double[oSums.Count];
            for (int i = 0; i < oSums.Count; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }
    }

    public abstract class AbstractNeuralNetwork
    {
        protected readonly Random Rnd;

        // ReSharper disable once FieldCanBeMadeReadOnly.Local
        protected int LayerCount;

        // ReSharper disable once FieldCanBeMadeReadOnly.Local
        protected int[] Sizes;

        /**
         * Values for layers
         */
        protected double[][] Layers;

        protected WeightComposite[] Neurons;

        public const double EtaPlus = 1.2; // values are from the paper
        public const double EtaMinus = 0.5;
        public const double DeltaMax = 50.0;
        public const double DeltaMin = 1.0E-6;

        public bool multiThread;
        public int threadCount;

        protected AbstractNeuralNetwork(IReadOnlyList<int> sizes)
        {
            this.LayerCount = sizes.Count;
            this.Sizes = new int[sizes.Count];
            for (int i = 0; i < sizes.Count; i++)
            {
                this.Sizes[i] = sizes[i];
            }
            this.Layers = new double[LayerCount][];
            this.Neurons = new WeightComposite[LayerCount];
            for (int i = 0; i < this.LayerCount; i++)
            {
                this.Layers[i] = new double[this.Sizes[i]];
            }
            for (int i = 1; i < this.LayerCount; i++)
            {
                this.Neurons[i].Biases = new double[this.Sizes[i]];
                this.Neurons[i].Weights = MakeMatrix(this.Sizes[i], this.Sizes[i - 1], 0.0);
            }

            this.Rnd = new Random();
            this.InitializeWeights(); // all weights and biases
        } // ctor

        protected abstract double[] ActivateFunction(IReadOnlyList<double> rawValues, int layerId);

        protected abstract double[][] CalculateGradTerms(double[][] rawValues,
            IReadOnlyList<double> tValues);

        protected static double[][] MakeMatrix(int rows,
            int cols, double v) // helper for ctor, Train
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result[i][j] = v;
            return result;
        }

        protected static double[] MakeVector(int len, double v) // helper for Train
        {
            double[] result = new double[len];
            for (int i = 0; i < len; ++i)
                result[i] = v;
            return result;
        }

        protected void InitializeWeights() // helper for ctor
        {
            for (int layer = 1; layer < this.LayerCount; layer++)
            {
                int size = this.Sizes[layer];
                int prev_size = this.Sizes[layer - 1];
                for (int node = 0; node < size; node++)
                {
                    this.Neurons[layer].Biases[node] = (0.001 - 0.0001) * Rnd.NextDouble() + 0.0001;
                    this.Neurons[layer].Weights[node] = new double[prev_size];
                    double vj = 0;
                    for (int i = 0; i < prev_size; i++)
                    {
                        this.Neurons[layer].Weights[node][i] = (0.001 - 0.0001) * Rnd.NextDouble() + 0.0001;
                        vj += Math.Pow(this.Neurons[layer].Weights[node][i], 2);
                    }
                    vj = 0.7 * Math.Pow(size, 1.0 / prev_size) / Math.Sqrt(vj);
                    for (int i = 0; i < prev_size; i++)
                    {
                        this.Neurons[layer].Weights[node][i] *= vj;
                    }
                }
            }
        }

        // ReSharper disable once InconsistentNaming
        public double[] TrainRPROP(double[][] trainData, int maxEpochs, double[][] testData) // using RPROP
        {
            WeightComposite[] allGradsAcc = new WeightComposite[this.LayerCount];
            WeightComposite[] prevGradsAcc = new WeightComposite[this.LayerCount];
            WeightComposite[] prevDeltas = new WeightComposite[this.LayerCount];
            for (int i = 1; i < this.LayerCount; i++)
            {
                int size = this.Sizes[i];
                int prevSize = this.Sizes[i - 1];

                // accumulated over all training data
                allGradsAcc[i].Biases = new double[size];
                allGradsAcc[i].Weights = MakeMatrix(size, prevSize, 0.0);

                // accumulated, previous iteration
                prevGradsAcc[i].Biases = new double[size];
                prevGradsAcc[i].Weights = MakeMatrix(size, prevSize, 0.0);

                // must save previous weight deltas
                prevDeltas[i].Biases = MakeVector(size, 0.01);
                prevDeltas[i].Weights = MakeMatrix(size, prevSize, 0.01);
            }

            if (this.multiThread && (this.threadCount == 0))
            {
                this.threadCount = Environment.ProcessorCount - 1;
            }

            {
                double[] currWts = this.GetWeights();
                double[] err = RootMeanSquaredError(trainData, currWts);
                double[] err_t = RootMeanSquaredError(testData, currWts);
                Console.WriteLine("\nepoch = pre; err = {0:F4} [{1:F4}]\ttest err = {2:F4} [{3:F4}]",
                    err[0], err[1], err_t[0], err_t[1]);
            }

            int epoch = 0;
            Stopwatch timer1 = new Stopwatch();
            Stopwatch timer2 = new Stopwatch();
            Stopwatch timer3 = new Stopwatch();
            timer3.Start();
            while (epoch < maxEpochs)
            {
                ++epoch;

                timer3.Stop();
                timer1.Start();
                // 1. compute and accumulate all gradients
                for (int layer = 1; layer < this.LayerCount; layer++)
                {
                    // zero-out values from prev iteration
                    ZeroOut(allGradsAcc[layer].Weights);
                    ZeroOut(allGradsAcc[layer].Biases);
                }
                double[] err = this.ComputeGraduate(trainData, allGradsAcc);
                // update all weights and biases (in any order)
                this.UpdateWeigtsAndBiases(allGradsAcc, prevGradsAcc, prevDeltas);
                timer1.Stop();

                timer3.Start();
                Console.Write(".");
                if ((epoch % 10 == 0) || (err[0] <= 0.0001))
                {
                    timer3.Stop();
                    timer2.Start();
                    double[] currWts = this.GetWeights();
                    double[] err_t = RootMeanSquaredError(testData, currWts);
                    Console.WriteLine("\nepoch = {0} err = {1:F4} [{2:F4}]\ttest err = {3:F4} [{4:F4}]",
                        epoch, err[0], err[1], err_t[0], err_t[1]);
                    timer2.Stop();
                    timer3.Start();
                    this.Save($"epoch-{epoch}.dat");
                    if (err[0] <= 0.001)
                    {
                        break;
                    }
                }
                else
                {
                    if (epoch % 5 == 0)
                    {
                        Console.Write(" ");
                    }
                }
            } // while
            timer3.Stop();
            Console.WriteLine("Elapsed time. Neuro = {0}, RMSE calculation = {1}, Other work = {2}",
                timer1.ElapsedMilliseconds / 1000, timer2.ElapsedMilliseconds / 1000,
                timer3.ElapsedMilliseconds / 1000);

            double[] wts = this.GetWeights();
            return wts;
        } // Train

        protected static void ZeroOut(double[][] matrix)
        {
            foreach (double[] t in matrix)
            {
                for (int j = 0; j < t.Length; ++j)
                {
                    t[j] = 0.0;
                }
            }
        }

        protected static void ZeroOut(double[] array) // helper for Train
        {
            for (int i = 0; i < array.Length; ++i)
                array[i] = 0.0;
        }

        /**
         * WeightsCount
         */
        public int GetWeightsCount()
        {
            int numWeights = 0;
            for (int layerNum = 1; layerNum < this.LayerCount; layerNum++)
            {
                numWeights += (this.Sizes[layerNum - 1] + 1) * this.Sizes[layerNum];
            }

            return numWeights;
        }

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = this.GetWeightsCount();
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0; // points into weights param
            for (int layerNum = 1; layerNum < this.LayerCount; layerNum++)
            {
                for (int i = 0; i < this.Sizes[layerNum]; ++i)
                {
                    for (int j = 0; j < this.Sizes[layerNum - 1]; ++j)
                    {
                        this.Neurons[layerNum].Weights[i][j] = weights[k++];
                    }
                }
                for (int i = 0; i < this.Sizes[layerNum]; ++i)
                {
                    this.Neurons[layerNum].Biases[i] = weights[k++];
                }
            }
        }

        public double[] GetWeights()
        {
            int numWeights = this.GetWeightsCount();

            double[] result = new double[numWeights];
            int k = 0;
            for (int layerNum = 1; layerNum < this.LayerCount; layerNum++)
            {
                for (int i = 0; i < this.Sizes[layerNum]; ++i)
                {
                    for (int j = 0; j < this.Sizes[layerNum - 1]; ++j)
                    {
                        result[k++] = this.Neurons[layerNum].Weights[i][j];
                    }
                }
                for (int i = 0; i < this.Sizes[layerNum]; ++i)
                {
                    result[k++] = this.Neurons[layerNum].Biases[i];
                }
            }

            return result;
        }

        public double[] ComputeOutputs(double[] xValues, double[][] outputLayers = null)
        {
            double[][] field = outputLayers ?? this.Layers;
            field[0] = xValues;
            for (int layer = 1; layer < this.LayerCount; layer++)
            {
                field[layer] = new double[this.Sizes[layer]];
                Array.Copy(this.Neurons[layer].Biases, field[layer], this.Sizes[layer]);

                for (int j = 0; j < this.Sizes[layer]; ++j) // compute i-h sum of weights * inputs
                {
                    for (int i = 0; i < this.Sizes[layer - 1]; ++i)
                    {
                        field[layer][j] +=
                            field[layer - 1][i] * this.Neurons[layer].Weights[j][i]; // note +=
                    }
                }

                field[layer] = this.ActivateFunction(field[layer], layer);
            }

            return field[this.LayerCount - 1];
        }

        public double Accuracy(double[][] testData, double[] weights)
        {
            this.SetWeights(weights);
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            int lastLayerId = this.LayerCount - 1;
            double[] xValues = new double[this.Sizes[0]]; // inputs
            double[] tValues = new double[this.Sizes[lastLayerId]]; // targets

            foreach (double[] t in testData)
            {
                Array.Copy(t, xValues, this.Sizes[0]); // parse data into x-values and t-values
                Array.Copy(t, this.Sizes[0], tValues, 0, this.Sizes[lastLayerId]);
                var yValues = this.ComputeOutputs(xValues); // computed Y
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y, double epsilon)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (double) numCorrect / (numCorrect + numWrong); // ugly 2 - check for divide by zero
        }

        public double[] RootMeanSquaredError(double[][] trainData, double[] weights)
        {
            return this.multiThread
                ? this.RootMeanSquaredErrorMultiThread(trainData, weights)
                : this.RootMeanSquaredErrorSingleThread(trainData, weights);
        }

        public double[] RootMeanSquaredErrorSingleThread(double[][] trainData, double[] weights)
        {
            this.SetWeights(weights); // copy the weights to evaluate in

            int lastLayerId = this.LayerCount - 1;
            int outputSize = this.Sizes[lastLayerId];
            int trainDataSize = trainData.Length;
            double[] xValues = new double[this.Sizes[0]]; // inputs
            double[] tValues = new double[outputSize]; // targets
            double sumSquaredError = 0.0;
            double sumSquaredErrorItem = 0.0;
            foreach (double[] t in trainData)
            {
                // following assumes data has all x-values first, followed by y-values!
                Array.Copy(t, xValues, this.Sizes[0]); // extract inputs
                Array.Copy(t, this.Sizes[0], tValues, 0, outputSize); // extract targets
                double[] yValues = this.ComputeOutputs(xValues);
                for (int j = 0; j < outputSize; ++j)
                {
                    double err = Math.Pow(yValues[j] - tValues[j], 2);
                    sumSquaredError += err / trainDataSize;
                    sumSquaredErrorItem += err / trainDataSize / outputSize;
                }
            }
            double[] d = {Math.Sqrt(sumSquaredErrorItem), Math.Sqrt(sumSquaredError)};
            return d;
        }

        private static int MaxIndex(IReadOnlyList<double> vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Count; ++i)
            {
                if (vector[i] <= biggestVal) continue;
                biggestVal = vector[i];
                bigIndex = i;
            }
            return bigIndex;
        }

        public void Save(string filename)
        {
            FileStream fo = File.Open(filename, FileMode.Create);
            BinaryWriter writer = new BinaryWriter(fo);
            writer.Write(this.LayerCount);
            for (int i = 0; i < this.LayerCount; i++)
            {
                writer.Write(this.Sizes[i]);
            }
            double[] weights = this.GetWeights();
            for (int id = 1; id < weights.Length; id++)
            {
                writer.Write(weights[id]);
            }
            writer.Close();
            fo.Close();
        }

        /**
         * update all weights and biases
         */
        protected double[] ComputeGraduate(double[][] trainData, WeightComposite[] allGradsAcc)
        {
            return this.multiThread
                ? this.ComputeGraduateMultiThread(trainData, allGradsAcc)
                : this.ComputeGraduateSingleThread(trainData, allGradsAcc);
        }

        /**
         * update all weights and biases
         */
        protected double[] ComputeGraduateSingleThread(double[][] trainData, WeightComposite[] allGradsAcc)
        {
            int lastLayerId = this.LayerCount - 1;
            int outputSize = this.Sizes[lastLayerId];
            double[] xValues = new double[this.Sizes[0]]; // inputs
            double[] tValues = new double[outputSize]; // target values
            double[] sumSquaredErrors = {0, 0};
            foreach (double[] t in trainData)
            {
                // no need to visit in random order because all rows processed before any updates ('batch')
                Array.Copy(t, xValues, this.Sizes[0]); // get the inputs
                Array.Copy(t, this.Sizes[0], tValues, 0, outputSize); // get the target values
                // copy xValues in, compute outputs using curr weights (and store outputs internally)
                double[] yValues = this.ComputeOutputs(xValues);

                double[][] gradTerms = this.CalculateGradTerms(this.Layers, tValues);

                for (int layer = lastLayerId; layer > 0; layer--)
                {
                    // add input to h-o component to make h-o weight gradients, and accumulate
                    for (int j = 0; j < this.Sizes[layer]; ++j)
                    {
                        double grad = gradTerms[layer][j];
                        allGradsAcc[layer].Biases[j] += grad;

                        for (int i = 0; i < this.Sizes[layer - 1]; ++i)
                        {
                            grad = gradTerms[layer][j] * this.Layers[layer - 1][i];
                            allGradsAcc[layer].Weights[j][i] += grad;
                        }
                    }
                }

                for (int j = 0; j < outputSize; ++j)
                {
                    double err = Math.Pow(yValues[j] - tValues[j], 2);
                    sumSquaredErrors[0] += err / trainData.Length;
                    sumSquaredErrors[1] += err / trainData.Length / this.Sizes[this.LayerCount - 1];
                }
            }

            return sumSquaredErrors;
        }

        /**
         * Подсчитываем градиент в несколько потоков
         */
        protected double[] ComputeGraduateMultiThread(double[][] trainData, WeightComposite[] allGradsAcc)
        {
            TaskFactory taskFactory = new TaskFactory();
            Task[] tasks = new Task[this.threadCount];
            ThreadInputDatum[] threadInputData = new ThreadInputDatum[this.threadCount];
            for (int i = 0; i < this.threadCount; i++)
            {
                threadInputData[i].field = new double[this.LayerCount][];
                threadInputData[i].allGradsAcc = new WeightComposite[this.LayerCount];
                threadInputData[i].xValues = new double[this.Sizes[0]]; // inputs
                threadInputData[i].tValues = new double[this.Sizes[this.LayerCount - 1]]; // targets

                threadInputData[i].delim1 = 1.0 / trainData.Length;
                threadInputData[i].delim2 = 1.0 / trainData.Length / this.Sizes[this.LayerCount - 1];
                threadInputData[i].sumSquaredErrors = new double[] {0, 0};
                for (int j = 0; j < this.LayerCount; j++)
                {
                    threadInputData[i].field[j] = new double[this.Sizes[j]];

                    if (j <= 0) continue;
                    threadInputData[i].allGradsAcc[j].Biases = new double[this.Sizes[j]];
                    threadInputData[i].allGradsAcc[j].Weights = MakeMatrix(this.Sizes[j], this.Sizes[j - 1], 0.0);
                }
            }

            List<double[]> innerTrainData = new List<double[]>(trainData);
            List<double[]> innerTrainDataChunk = new List<double[]>();
            int chunk_size = (int) (innerTrainData.Count * 0.8 / this.threadCount);
            while (innerTrainData.Count > 0)
            {
                int currentThread = -1;

                for (int i = 0; i < this.threadCount; i++)
                {
                    // ReSharper disable once InvertIf
                    if ((tasks[i] == null) || tasks[i].IsCompleted)
                    {
                        currentThread = i;
                        break;
                    }
                }

                if (currentThread == -1)
                {
                    Thread.Sleep(20);
                    continue;
                }

                innerTrainDataChunk.Clear();
                while ((innerTrainDataChunk.Count < chunk_size) && (innerTrainData.Count > 0))
                {
                    innerTrainDataChunk.Add(innerTrainData[0]);
                    innerTrainData.RemoveAt(0);
                }
                threadInputData[currentThread].trainDatum = innerTrainDataChunk.ToArray();

                for (int layer = 1; layer < this.LayerCount; layer++)
                {
                    // zero-out values from prev iteration
                    ZeroOut(threadInputData[currentThread].allGradsAcc[layer].Weights);
                    ZeroOut(threadInputData[currentThread].allGradsAcc[layer].Biases);
                }

                tasks[currentThread] =
                    taskFactory.StartNew(this.ComputeGraduateInThread, threadInputData[currentThread]);
            }

            for (int i = 0; i < this.threadCount; i++)
            {
                if (tasks[i] != null)
                {
                    tasks[i].Wait();
                }
            }

            // Всё в allGradsAcc
            for (int i = 0; i < this.threadCount; i++)
            {
                for (int layer = 1; layer < this.LayerCount; layer++)
                {
                    for (int size = 0; size < this.Sizes[layer]; size++)
                    {
                        allGradsAcc[layer].Biases[size] += threadInputData[i].allGradsAcc[layer].Biases[size];
                        for (int prev_size = 0; prev_size < this.Sizes[layer - 1]; prev_size++)
                        {
                            allGradsAcc[layer].Weights[size][prev_size]
                                += threadInputData[i].allGradsAcc[layer].Weights[size][prev_size];
                        }
                    }
                }
            }

            double sumSquaredErrorItem = 0;
            double sumSquaredError = 0;
            for (int i = 0; i < this.threadCount; i++)
            {
                sumSquaredError += threadInputData[i].sumSquaredErrors[0];
                sumSquaredErrorItem += threadInputData[i].sumSquaredErrors[1];
            }

            double[] d = {Math.Sqrt(sumSquaredErrorItem), Math.Sqrt(sumSquaredError)};
            return d;
        }

        public void ComputeGraduateInThread(object input)
        {
            ThreadInputDatum inputDatum = (ThreadInputDatum) input;

            int lastLayerId = this.LayerCount - 1;

            foreach (double[] t in inputDatum.trainDatum)
            {
                // no need to visit in random order because all rows processed before any updates ('batch')
                Array.Copy(t, inputDatum.xValues, this.Sizes[0]); // get the inputs
                // get the target values
                Array.Copy(t, this.Sizes[0], inputDatum.tValues, 0, this.Sizes[lastLayerId]);
                // copy xValues in, compute outputs using curr weights (and store outputs internally)
                double[] yValues = this.ComputeOutputs(inputDatum.xValues, inputDatum.field);

                double[][] gradTerms = this.CalculateGradTerms(inputDatum.field, inputDatum.tValues);

                for (int layer = lastLayerId; layer > 0; layer--)
                {
                    // add input to h-o component to make h-o weight gradients, and accumulate
                    for (int j = 0; j < this.Sizes[layer]; ++j)
                    {
                        double grad = gradTerms[layer][j];
                        inputDatum.allGradsAcc[layer].Biases[j] += grad;

                        for (int i = 0; i < this.Sizes[layer - 1]; ++i)
                        {
                            grad = gradTerms[layer][j] * inputDatum.field[layer - 1][i];
                            inputDatum.allGradsAcc[layer].Weights[j][i] += grad;
                        }
                    }
                }

                for (int j = 0; j < this.Sizes[lastLayerId]; ++j)
                {
                    double err = Math.Pow(yValues[j] - inputDatum.tValues[j], 2);
                    inputDatum.sumSquaredErrors[0] += err * inputDatum.delim1;
                    inputDatum.sumSquaredErrors[1] += err * inputDatum.delim2;
                }
            }
        }

        public double[] RootMeanSquaredErrorMultiThread(double[][] trainData, double[] weights)
        {
            this.SetWeights(weights); // copy the weights to evaluate in
            TaskFactory taskFactory = new TaskFactory();
            Task[] tasks = new Task[this.threadCount];
            int lastLayerId = this.LayerCount - 1;
            int outputSize = this.Sizes[lastLayerId];

            RMSEThreadInputDatum[] threadInputData = new RMSEThreadInputDatum[this.threadCount];
            for (int i = 0; i < this.threadCount; i++)
            {
                threadInputData[i].field = new double[this.LayerCount][];
                threadInputData[i].xValues = new double[this.Sizes[0]]; // inputs
                threadInputData[i].tValues = new double[outputSize]; // targets
                threadInputData[i].delim1 = 1.0 / trainData.Length;
                threadInputData[i].delim2 = 1.0 / trainData.Length / outputSize;
                threadInputData[i].sumSquaredErrors = new double[2];
                for (int j = 0; j < this.LayerCount; j++)
                {
                    threadInputData[i].field[j] = new double[this.Sizes[j]];
                }
            }

            List<double[]> innerTrainData = new List<double[]>(trainData);
            List<double[]> innerTrainDataChunk = new List<double[]>();
            int chunk_size = (int) (innerTrainData.Count * 0.8 / this.threadCount);
            while (innerTrainData.Count > 0)
            {
                int currentThread = -1;

                for (int i = 0; i < this.threadCount; i++)
                {
                    // ReSharper disable once InvertIf
                    if ((tasks[i] == null) || tasks[i].IsCompleted)
                    {
                        currentThread = i;
                        break;
                    }
                }

                if (currentThread == -1)
                {
                    Thread.Sleep(20);
                    continue;
                }

                innerTrainDataChunk.Clear();
                while ((innerTrainDataChunk.Count < chunk_size) && (innerTrainData.Count > 0))
                {
                    innerTrainDataChunk.Add(innerTrainData[0]);
                    innerTrainData.RemoveAt(0);
                }
                threadInputData[currentThread].trainDatum = innerTrainDataChunk.ToArray();

                tasks[currentThread] =
                    taskFactory.StartNew(this.ComputeRMSEInThread, threadInputData[currentThread]);
            }

            for (int i = 0; i < this.threadCount; i++)
            {
                if (tasks[i] != null)
                {
                    tasks[i].Wait();
                }
            }

            double sumSquaredErrorItem = 0;
            double sumSquaredError = 0;
            for (int i = 0; i < this.threadCount; i++)
            {
                sumSquaredError += threadInputData[i].sumSquaredErrors[0];
                sumSquaredErrorItem += threadInputData[i].sumSquaredErrors[1];
            }

            double[] d = {Math.Sqrt(sumSquaredErrorItem), Math.Sqrt(sumSquaredError)};
            return d;
        }

        public void ComputeRMSEInThread(object input)
        {
            RMSEThreadInputDatum threadInputDatum = (RMSEThreadInputDatum) input;
            int outputSize = this.Sizes[this.LayerCount - 1];

            foreach (double[] t in threadInputDatum.trainDatum)
            {
                // following assumes data has all x-values first, followed by y-values!
                Array.Copy(t, threadInputDatum.xValues, this.Sizes[0]); // extract inputs
                Array.Copy(t, this.Sizes[0], threadInputDatum.tValues, 0, outputSize); // extract targets
                double[] yValues = this.ComputeOutputs(threadInputDatum.xValues, threadInputDatum.field);
                for (int j = 0; j < outputSize; ++j)
                {
                    double err = Math.Pow(yValues[j] - threadInputDatum.tValues[j], 2);
                    threadInputDatum.sumSquaredErrors[0] += err * threadInputDatum.delim1;
                    threadInputDatum.sumSquaredErrors[1] += err * threadInputDatum.delim2;
                }
            }
        }

        protected void UpdateWeigtsAndBiases(WeightComposite[] allGradsAcc,
            WeightComposite[] prevGradsAcc,
            WeightComposite[] prevDeltas)
        {
            // update input-hidden weights
            for (int layer = 1; layer < this.LayerCount; layer++)
            {
                int size = this.Sizes[layer];
                int previousSize = this.Sizes[layer - 1];

                for (int i = 0; i < previousSize; ++i)
                {
                    for (int j = 0; j < size; ++j)
                    {
                        double delta = prevDeltas[layer].Weights[j][i];
                        double t = prevGradsAcc[layer].Weights[j][i] * allGradsAcc[layer].Weights[j][i];
                        if (t > 0) // no sign change, increase delta
                        {
                            delta *= AbstractNeuralNetwork.EtaPlus; // compute delta
                            // keep it in range
                            if (delta > AbstractNeuralNetwork.DeltaMax)
                            {
                                delta = AbstractNeuralNetwork.DeltaMax;
                            }
                            // determine direction and magnitude
                            double tmp = -Math.Sign(allGradsAcc[layer].Weights[j][i]) * delta;
                            this.Neurons[layer].Weights[j][i] += tmp; // update weights
                        }
                        else if (t < 0) // grad changed sign, decrease delta
                        {
                            delta *= AbstractNeuralNetwork.EtaMinus; // the delta (not used, but saved for later)
                            // keep it in range
                            if (delta < AbstractNeuralNetwork.DeltaMin)
                            {
                                delta = AbstractNeuralNetwork.DeltaMin;
                            }
                            this.Neurons[layer].Weights[j][i] -=
                                prevDeltas[layer].Weights[j][i]; // revert to previous weight
                            allGradsAcc[layer].Weights[j][i] = 0; // forces next if-then branch, next iteration
                        }
                        else // this happens next iteration after 2nd branch above (just had a change in gradient)
                        {
                            // no change to delta
                            // no way should delta be 0 ...
                            double tmp = -Math.Sign(allGradsAcc[layer].Weights[j][i]) * delta; // determine direction
                            this.Neurons[layer].Weights[j][i] += tmp; // update
                        }

                        prevDeltas[layer].Weights[j][i] = delta; // save delta
                        // save the (accumulated) gradient
                        prevGradsAcc[layer].Weights[j][i] = allGradsAcc[layer].Weights[j][i];
                    } // j
                } // i

                // update (input-to-) hidden biases
                for (int i = 0; i < size; ++i)
                {
                    double delta = prevDeltas[layer].Biases[i];
                    double t = prevGradsAcc[layer].Biases[i] * allGradsAcc[layer].Biases[i];
                    if (t > 0) // no sign change, increase delta
                    {
                        delta *= AbstractNeuralNetwork.EtaPlus; // compute delta
                        if (delta > AbstractNeuralNetwork.DeltaMax)
                        {
                            delta = AbstractNeuralNetwork.DeltaMax;
                        }
                        double tmp = -Math.Sign(allGradsAcc[layer].Biases[i]) * delta; // determine direction
                        this.Neurons[layer].Biases[i] += tmp; // update
                    }
                    else if (t < 0) // grad changed sign, decrease delta
                    {
                        delta *= AbstractNeuralNetwork.EtaMinus; // the delta (not used, but saved later)
                        if (delta < AbstractNeuralNetwork.DeltaMin)
                        {
                            delta = AbstractNeuralNetwork.DeltaMin;
                        }
                        this.Neurons[layer].Biases[i] -= prevDeltas[layer].Biases[i]; // revert to previous weight
                        allGradsAcc[layer].Biases[i] = 0; // forces next branch, next iteration
                    }
                    else // this happens next iteration after 2nd branch above (just had a change in gradient)
                    {
                        if (delta > AbstractNeuralNetwork.DeltaMax)
                        {
                            delta = AbstractNeuralNetwork.DeltaMax;
                        }
                        else if (delta < AbstractNeuralNetwork.DeltaMin)
                        {
                            delta = AbstractNeuralNetwork.DeltaMin;
                        }
                        // no way should delta be 0 . . .
                        double tmp = -Math.Sign(allGradsAcc[layer].Biases[i]) * delta; // determine direction
                        this.Neurons[layer].Biases[i] += tmp; // update
                    }
                    prevDeltas[layer].Biases[i] = delta;
                    prevGradsAcc[layer].Biases[i] = allGradsAcc[layer].Biases[i];
                }
            }
        }
    } // NeuralNetwork
} // ns