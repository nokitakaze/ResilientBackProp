using System;
using System.IO;

// See "A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm",
// M. Riedmiller and H. Braun,
// Proceedings of the 1993 IEEE International Conference on Neural Networks,
// pp. 586-591
// This is the orginal version of the algorithm. There are many later variations.

namespace ResilientBackProp
{
    class RpropProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network with Resilient Back-Propagation (RPROP) training demo");

            int numInput = 4; // number features
            int numHidden = 5;
            int numOutput = 3; // number of classes for Y
            int numRows = 10000;

            Console.WriteLine("\nGenerating " + numRows +
              " artificial data items with " + numInput + " features");
            double[][] allData = MakeAllData(numInput, numHidden, numOutput, numRows);
            Console.WriteLine("Done");

            Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, 0.80, out trainData, out testData);
            Console.WriteLine("Done");

            Console.WriteLine("\nTraining data: \n");
            ShowData(trainData, 4, 2, true);

            Console.WriteLine("Test data: \n");
            ShowData(testData, 3, 2, true);

            Console.WriteLine("Creating a 4-5-3 neural network");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);
            nn.save("before_test.dat");

            int maxEpochs = 1000;
            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);

            Console.WriteLine("\nStarting RPROP training");
            double[] weights = nn.TrainRPROP(trainData, maxEpochs); // RPROP
            nn.save("after_test.dat");
            Console.WriteLine("Done");
            Console.WriteLine("\nFinal neural network model weights:\n");
            ShowVector(weights, 4, 10, true);

            double trainAcc = nn.Accuracy(trainData, weights);
            Console.WriteLine("\nAccuracy on training data = " +
              trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData, weights);
            Console.WriteLine("\nAccuracy on test data = " +
              testAcc.ToString("F4"));

            Console.WriteLine("\nEnd neural network with Resilient Propagation demo\n");
            Console.ReadLine();
        } // Main

        /**
         * Generate synthetic data
         */
        static double[][] MakeAllData(int numInput, int numHidden, int numOutput,
          int numRows)
        {
            Random rnd = new Random();
            int numWeights = (numInput * numHidden) + numHidden +
              (numHidden * numOutput) + numOutput;
            double[] weights = new double[numWeights]; // actually weights & biases
            for (int i = 0; i < numWeights; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to -10.0]

            Console.WriteLine("Generating weights:");
            ShowVector(weights, 4, 10, true);

            double[][] result = new double[numRows][]; // allocate return-result matrix
            for (int i = 0; i < numRows; ++i)
                result[i] = new double[numInput + numOutput]; // 1-of-N Y in last column

            NeuralNetwork gnn =
              new NeuralNetwork(numInput, numHidden, numOutput); // generating NN
            gnn.SetWeights(weights);

            for (int r = 0; r < numRows; ++r) // for each row
            {
                // generate random inputs
                double[] inputs = new double[numInput];
                for (int i = 0; i < numInput; ++i)
                    inputs[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to -10.0]

                // compute outputs
                double[] outputs = gnn.ComputeOutputs(inputs);
                // translate outputs to 1-of-N
                double[] oneOfN = new double[numOutput]; // all 0.0
                int maxIndex = 0;
                double maxValue = outputs[0];
                for (int i = 0; i < numOutput; ++i)
                {
                    if (outputs[i] > maxValue)
                    {
                        maxIndex = i;
                        maxValue = outputs[i];
                    }
                }
                oneOfN[maxIndex] = 1.0;

                // place inputs and 1-of-N output values into curr row
                int c = 0; // column into result[][]
                for (int i = 0; i < numInput; ++i) // inputs
                    result[r][c++] = inputs[i];
                for (int i = 0; i < numOutput; ++i) // outputs
                    result[r][c++] = oneOfN[i];
            } // each row
            return result;
        } // MakeAllData

        /**
         * Put synthetic data to train and test
         */
        static void MakeTrainTest(double[][] allData, double trainPct,
          out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random();
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of all data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } // MakeTrainTest

        public static void ShowData(double[][] data, int numRows,
          int decimals, bool indices)
        {
            int len = data.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
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
            if (indices == true)
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
            if (newLine == true)
                Console.WriteLine("");
        }

    } // Program

    public class NeuralNetwork
    {
        private Random rnd;
        private int layer_count = 3;
        int[] sizes;

        private double[][] layers;
        private double[][] biases;
        private double[][][] weights;

        const double etaPlus = 1.2; // values are from the paper
        const double etaMinus = 0.5;
        const double deltaMax = 50.0;
        const double deltaMin = 1.0E-6;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.sizes = new int[layer_count];
            this.sizes[0] = numInput;
            this.sizes[1] = numHidden;
            this.sizes[2] = numOutput;
            this.layers = new double[layer_count][];
            this.biases = new double[layer_count][];
            this.weights = new double[layer_count][][];
            for (int i = 0; i < this.layer_count; i++)
            {
                this.layers[i] = new double[this.sizes[i]];
            }
            for (int i = 1; i < this.layer_count; i++)
            {
                this.biases[i] = new double[this.sizes[i]];
                this.weights[i] = MakeMatrix(this.sizes[i - 1], this.sizes[i], 0.0);
            }

            this.rnd = new Random();
            this.InitializeWeights(); // all weights and biases
        } // ctor

        private static double[][] MakeMatrix(int rows,
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

        private static double[] MakeVector(int len, double v) // helper for Train
        {
            double[] result = new double[len];
            for (int i = 0; i < len; ++i)
                result[i] = v;
            return result;
        }

        private void InitializeWeights() // helper for ctor
        {
            // initialize weights and biases to random values between 0.0001 and 0.001
            int numWeights = (this.sizes[0] * this.sizes[1]) +
                (this.sizes[1] * this.sizes[2]) +
                this.sizes[1] + this.sizes[2];
            double[] initialWeights = new double[numWeights];
            // @todo Переписать, веса надо нормализировать
            for (int i = 0; i < initialWeights.Length; ++i)
            {
                initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            }
            this.SetWeights(initialWeights);
        }

        public double[] TrainRPROP(double[][] trainData, int maxEpochs) // using RPROP
        {
            // there is an accumulated gradient and a previous gradient for each weight and bias
            double[][] GradTerms = new double[this.layer_count][];
            double[][] BiasGradsAcc = new double[this.layer_count][];
            double[][] PrevBiasGradsAcc = new double[this.layer_count][];
            double[][] PrevBiasDeltas = new double[this.layer_count][];

            double[][][] WeightGradsAcc = new double[this.layer_count][][];
            double[][][] PrevWeightGradsAcc = new double[this.layer_count][][];
            double[][][] PrevWeightDeltas = new double[this.layer_count][][];
            for (int i = 1; i < this.layer_count; i++)
            {
                GradTerms[i] = new double[this.sizes[i]];
                BiasGradsAcc[i] = new double[this.sizes[i]];
                PrevBiasGradsAcc[i] = new double[this.sizes[i]];
                PrevBiasDeltas[i] = MakeVector(this.sizes[i], 0.01);

                // accumulated over all training data
                WeightGradsAcc[i] = MakeMatrix(this.sizes[i - 1], this.sizes[i], 0.0);

                // accumulated, previous iteration
                PrevWeightGradsAcc[i] = MakeMatrix(this.sizes[i - 1], this.sizes[i], 0.0);

                // must save previous weight deltas
                PrevWeightDeltas[i] = MakeMatrix(this.sizes[i - 1], this.sizes[i], 0.01);
            }

            int epoch = 0;
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % 100 == 0 && epoch != maxEpochs)
                {
                    double[] currWts = this.GetWeights();
                    double[] err = RootMeanSquaredError(trainData, currWts);
                    Console.WriteLine("epoch = " + epoch + " err = " + err[0].ToString("F4") + " [" + err[1].ToString("F4") + "]");
                }

                // 1. compute and accumulate all gradients
                for (int layer = 1; layer < this.layer_count; layer++)
                {
                    // zero-out values from prev iteration
                    ZeroOut(WeightGradsAcc[layer]);
                    ZeroOut(BiasGradsAcc[layer]);
                }
                this.ComputeGraduate(trainData, GradTerms, WeightGradsAcc, BiasGradsAcc);
                // update all weights and biases (in any order)
                this.UpdateWeigtsAndBiases(PrevWeightGradsAcc, WeightGradsAcc, PrevWeightDeltas,
                    PrevBiasGradsAcc, BiasGradsAcc, PrevBiasDeltas);
            } // while

            double[] wts = this.GetWeights();
            return wts;
        } // Train

        private static void ZeroOut(double[][] matrix)
        {
            for (int i = 0; i < matrix.Length; ++i)
                for (int j = 0; j < matrix[i].Length; ++j)
                    matrix[i][j] = 0.0;
        }

        private static void ZeroOut(double[] array) // helper for Train
        {
            for (int i = 0; i < array.Length; ++i)
                array[i] = 0.0;
        }

        public void SetWeights(double[] weights)
        {
            // @todo переписать на layer_count
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (this.sizes[0] * this.sizes[1]) + (this.sizes[1] * this.sizes[2]) +
                this.sizes[1] + this.sizes[2];
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0; // points into weights param

            for (int i = 0; i < this.sizes[0]; ++i)
                for (int j = 0; j < this.sizes[1]; ++j)
                    this.weights[1][i][j] = weights[k++];
            for (int i = 0; i < this.sizes[1]; ++i)
                this.biases[1][i] = weights[k++];
            for (int i = 0; i < this.sizes[1]; ++i)
                for (int j = 0; j < this.sizes[2]; ++j)
                    this.weights[2][i][j] = weights[k++];
            for (int i = 0; i < this.sizes[2]; ++i)
                this.biases[2][i] = weights[k++];
        }

        public double[] GetWeights()
        {
            // @todo переписать на layer_count
            int numWeights = (this.sizes[0] * this.sizes[1]) + (this.sizes[1] * this.sizes[2]) +
                this.sizes[1] + this.sizes[2];
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < this.weights[1].Length; ++i)
                for (int j = 0; j < this.weights[1][0].Length; ++j)
                    result[k++] = this.weights[1][i][j];
            for (int i = 0; i < this.biases[1].Length; ++i)
                result[k++] = this.biases[1][i];
            for (int i = 0; i < this.weights[2].Length; ++i)
                for (int j = 0; j < this.weights[2][0].Length; ++j)
                    result[k++] = this.weights[2][i][j];
            for (int i = 0; i < this.biases[2].Length; ++i)
                result[k++] = this.biases[2][i];
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            this.layers[0] = xValues;
            for (int layer = 1; layer < this.layer_count; layer++)
            {
                this.layers[layer] = new double[this.sizes[layer]];
                Array.Copy(this.biases[layer], this.layers[layer], this.sizes[layer]);

                for (int j = 0; j < this.sizes[layer]; ++j)  // compute i-h sum of weights * inputs
                {
                    for (int i = 0; i < this.sizes[layer - 1]; ++i)
                    {
                        this.layers[layer][j] += this.layers[layer - 1][i] * this.weights[layer][i][j]; // note +=
                    }
                }

                if (layer < this.layer_count - 1)
                {
                    for (int i = 0; i < this.sizes[1]; ++i)   // apply activation
                    {
                        this.layers[layer][i] = HyperTan(this.layers[layer][i]); // hard-coded
                    }
                }
                else
                {
                    double[] softOut = Softmax(this.layers[layer]); // softmax activation does all outputs at once for efficiency
                    Array.Copy(softOut, this.layers[layer], softOut.Length);
                }
            }

            double[] retResult = new double[this.sizes[this.layer_count - 1]]; // could define a GetOutputs method instead
            Array.Copy(this.layers[this.layer_count - 1], retResult, retResult.Length);
            return retResult;
        }

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // does all output nodes at once so scale doesn't have to be re-computed each time
            // determine max output-sum
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }

        public double Accuracy(double[][] testData, double[] weights)
        {
            // @todo переписать на layer_count
            this.SetWeights(weights);
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[this.sizes[0]]; // inputs
            double[] tValues = new double[this.sizes[2]]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, this.sizes[0]); // parse data into x-values and t-values
                Array.Copy(testData[i], this.sizes[0], tValues, 0, this.sizes[2]);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y, double epsilon)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
        }

        public double[] RootMeanSquaredError(double[][] trainData, double[] weights)
        {
            // @todo переписать на layer_count
            this.SetWeights(weights); // copy the weights to evaluate in

            double[] xValues = new double[this.sizes[0]]; // inputs
            double[] tValues = new double[this.sizes[2]]; // targets
            double sumSquaredError = 0.0;
            double sumSquaredError_item = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // walk through each training data item
            {
                // following assumes data has all x-values first, followed by y-values!
                Array.Copy(trainData[i], xValues, this.sizes[0]); // extract inputs
                Array.Copy(trainData[i], this.sizes[0], tValues, 0, this.sizes[2]); // extract targets
                double[] yValues = this.ComputeOutputs(xValues);
                double this_item_err_sum = 0;
                for (int j = 0; j < yValues.Length; ++j)
                {
                    double err = Math.Pow(yValues[j] - tValues[j], 2);
                    sumSquaredError += err / trainData.Length;
                    this_item_err_sum += err / yValues.Length;
                }
                sumSquaredError_item += this_item_err_sum / trainData.Length;
            }
            double[] d = new double[2];
            d[0] = Math.Sqrt(sumSquaredError_item);
            d[1] = Math.Sqrt(sumSquaredError);
            return d;
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }

        public void save(string filename)
        {
            FileStream fo = File.Open(filename, FileMode.Create);
            BinaryWriter writer = new BinaryWriter(fo);
            writer.Write(this.layer_count);
            for (int i = 0; i < this.layer_count; i++)
            {
                writer.Write(this.sizes[i]);
            }
            for (int layer = 1; layer < this.layer_count; layer++)
            {
                int size = this.sizes[layer];
                int size_previous = this.sizes[layer - 1];
                for (int node = 0; node < size; node++)
                {
                    writer.Write(this.biases[layer][node]);
                }
                for (int node = 0; node < size; node++)
                {
                    for (int prev = 0; prev < size_previous; prev++)
                    {
                        writer.Write(this.weights[layer][prev][node]);
                    }
                }
            }
            writer.Close();
            fo.Close();
        }

        void ComputeGraduate(double[][] trainData, double[][] GradTerms, double[][][] WeightGradsAcc, double[][] BiasGradsAcc)
        {
            int last_layer_id = this.layer_count - 1;
            double[] xValues = new double[this.sizes[0]]; // inputs
            double[] tValues = new double[this.sizes[last_layer_id]]; // target values
            for (int row = 0; row < trainData.Length; ++row)  // walk thru all training data
            {
                // no need to visit in random order because all rows processed before any updates ('batch')
                Array.Copy(trainData[row], xValues, this.sizes[0]); // get the inputs
                Array.Copy(trainData[row], this.sizes[0], tValues, 0, this.sizes[2]); // get the target values
                ComputeOutputs(xValues); // copy xValues in, compute outputs using curr weights (and store outputs internally)

                // compute the h-o gradient term/component as in regular back-prop
                // this term usually is lower case Greek delta but there are too many other deltas below
                for (int i = 0; i < this.sizes[last_layer_id]; ++i)
                {
                    double value = this.layers[last_layer_id][i];
                    double derivative = (1 - value) * value; // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                    GradTerms[last_layer_id][i] = derivative * (value - tValues[i]); // careful with O-T vs. T-O, O-T is the most usual
                }

                // compute the i-h gradient term/component as in regular back-prop
                for (int i = 0; i < this.sizes[1]; ++i)
                {
                    double value = this.layers[1][i];
                    double derivative = (1 - value) * (1 + value); // derivative of tanh = (1 - y) * (1 + y)
                    double sum = 0.0;
                    for (int j = 0; j < this.sizes[last_layer_id]; ++j) // each hidden delta is the sum of this.sizes[2] terms
                    {
                        double x = GradTerms[last_layer_id][j] * this.weights[last_layer_id][i][j];
                        sum += x;
                    }
                    GradTerms[1][i] = derivative * sum;
                }

                for (int layer = this.layer_count - 1; layer > 0; layer--)
                {
                    // add input to h-o component to make h-o weight gradients, and accumulate
                    for (int j = 0; j < this.sizes[layer]; ++j)
                    {
                        double grad = GradTerms[layer][j];
                        BiasGradsAcc[layer][j] += grad;

                        for (int i = 0; i < this.sizes[layer - 1]; ++i)
                        {
                            grad = GradTerms[layer][j] * this.layers[layer - 1][i];
                            WeightGradsAcc[layer][i][j] += grad;
                        }
                    }
                }
            } // each row
        }

        void UpdateWeigtsAndBiases(double[][][] PrevWeightGradsAcc, double[][][] WeightGradsAcc, double[][][] PrevWeightDeltas,
            double[][] PrevBiasGradsAcc, double[][] BiasGradsAcc, double[][] PrevBiasDeltas)
        {
            // update input-hidden weights
            for (int layer = 1; layer < this.layer_count; layer++)
            {
                int size = this.sizes[layer];
                int previous_size = this.sizes[layer - 1];

                for (int i = 0; i < previous_size; ++i)
                {
                    for (int j = 0; j < size; ++j)
                    {
                        double delta = PrevWeightDeltas[layer][i][j];
                        double t = PrevWeightGradsAcc[layer][i][j] * WeightGradsAcc[layer][i][j];
                        if (t > 0) // no sign change, increase delta
                        {
                            delta *= etaPlus; // compute delta
                            if (delta > deltaMax) delta = deltaMax; // keep it in range
                            double tmp = -Math.Sign(WeightGradsAcc[layer][i][j]) * delta; // determine direction and magnitude
                            this.weights[layer][i][j] += tmp; // update weights
                        }
                        else if (t < 0) // grad changed sign, decrease delta
                        {
                            delta *= etaMinus; // the delta (not used, but saved for later)
                            if (delta < deltaMin) delta = deltaMin; // keep it in range
                            this.weights[layer][i][j] -= PrevWeightDeltas[layer][i][j]; // revert to previous weight
                            WeightGradsAcc[layer][i][j] = 0; // forces next if-then branch, next iteration
                        }
                        else // this happens next iteration after 2nd branch above (just had a change in gradient)
                        {
                            // no change to delta
                            // no way should delta be 0 . . . 
                            double tmp = -Math.Sign(WeightGradsAcc[layer][i][j]) * delta; // determine direction
                            this.weights[layer][i][j] += tmp; // update
                        }

                        PrevWeightDeltas[layer][i][j] = delta; // save delta
                        PrevWeightGradsAcc[layer][i][j] = WeightGradsAcc[layer][i][j]; // save the (accumulated) gradient
                    } // j
                } // i

                // update (input-to-) hidden biases
                for (int i = 0; i < size; ++i)
                {
                    double delta = PrevBiasDeltas[layer][i];
                    double t = PrevBiasGradsAcc[layer][i] * BiasGradsAcc[layer][i];
                    if (t > 0) // no sign change, increase delta
                    {
                        delta *= etaPlus; // compute delta
                        if (delta > deltaMax) delta = deltaMax;
                        double tmp = -Math.Sign(BiasGradsAcc[layer][i]) * delta; // determine direction
                        this.biases[layer][i] += tmp; // update
                    }
                    else if (t < 0) // grad changed sign, decrease delta
                    {
                        delta *= etaMinus; // the delta (not used, but saved later)
                        if (delta < deltaMin) delta = deltaMin;
                        this.biases[layer][i] -= PrevBiasDeltas[layer][i]; // revert to previous weight
                        BiasGradsAcc[layer][i] = 0; // forces next branch, next iteration
                    }
                    else // this happens next iteration after 2nd branch above (just had a change in gradient)
                    {
                        if (delta > deltaMax) delta = deltaMax;
                        else if (delta < deltaMin) delta = deltaMin;
                        // no way should delta be 0 . . . 
                        double tmp = -Math.Sign(BiasGradsAcc[layer][i]) * delta; // determine direction
                        this.biases[layer][i] += tmp; // update
                    }
                    PrevBiasDeltas[layer][i] = delta;
                    PrevBiasGradsAcc[layer][i] = BiasGradsAcc[layer][i];
                }
            }
        }
    } // NeuralNetwork

} // ns
