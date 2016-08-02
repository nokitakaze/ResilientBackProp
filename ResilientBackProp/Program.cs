using System;

namespace ResilientBackProp
{
    class Program
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

            int maxEpochs = 1000;
            Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);

            Console.WriteLine("\nStarting RPROP training");
            double[] weights = nn.TrainRPROP(trainData, maxEpochs); // RPROP
            Console.WriteLine("Done");
            Console.WriteLine("\nFinal neural network model weights:\n");
            ShowVector(weights, 4, 10, true);

            double trainAcc = nn.Accuracy(trainData, weights);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData, weights);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));

            Console.WriteLine("\nEnd neural network with Resilient Propagation demo\n");
            Console.ReadLine();
        } // Main

        static double[][] MakeAllData(int numInput, int numHidden, int numOutput, int numRows)
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

        static void MakeTrainTest(double[][] allData, double trainPct, out double[][] trainData, out double[][] testData)
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
        private double[][] values;
        private double[][] biases;
        private double[][][] weights;

        private Random rnd;
        const int layer_count = 3;
        int[] sizes;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.sizes = new int[NeuralNetwork.layer_count];
            this.sizes[0] = numInput;
            this.sizes[1] = numHidden;
            this.sizes[2] = numOutput;

            this.values = new double[NeuralNetwork.layer_count][];
            this.biases = new double[NeuralNetwork.layer_count][];
            this.weights = new double[NeuralNetwork.layer_count][][];
            for (int layer = 0; layer < NeuralNetwork.layer_count; layer++)
            {
                this.values[layer] = new double[this.sizes[layer]];
            }
            for (int layer = 1; layer < NeuralNetwork.layer_count; layer++)
            {
                this.biases[layer] = new double[this.sizes[layer]];
                this.weights[layer] = MakeMatrix(this.sizes[layer - 1], this.sizes[layer], 0.0);
            }

            this.rnd = new Random(0);
            this.InitializeWeights(); // all weights and biases
        } // ctor

        private static double[][] MakeMatrix(int rows, int cols, double v) // helper for ctor, Train
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
            int numWeights = (this.sizes[0] * this.sizes[1]) + (this.sizes[1] * this.sizes[NeuralNetwork.layer_count - 1]) + this.sizes[1] + this.sizes[NeuralNetwork.layer_count - 1];
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.SetWeights(initialWeights);
        }

        public double[] TrainRPROP(double[][] trainData, int maxEpochs) // using RPROP
        {
            const double etaPlus = 1.2; // values are from the paper
            const double etaMinus = 0.5;
            const double deltaMax = 50.0;
            const double deltaMin = 1.0E-6;

            // Переписываем
            double[][] allGradTerms = new double[layer_count][];
            // there is an accumulated gradient and a previous gradient for each weight and bias
            double[][][] allWeightGradsAcc = new double[layer_count][][];
            double[][] allBiasGradsAcc = new double[layer_count][];
            // must save previous weight deltas
            double[][][] allPrevWeightGradsAcc = new double[layer_count][][];
            double[][] allPrevBiasGradsAcc = new double[layer_count][];
            // must save previous weight deltas
            double[][][] allPrevWeightDeltas = new double[layer_count][][];
            double[][] allPrevBiasDeltas = new double[layer_count][];
            for (int layer = 1; layer < layer_count; layer++)
            {
                int size = sizes[layer];
                int prev_size = sizes[layer - 1];
                allGradTerms[layer] = new double[size];

                allWeightGradsAcc[layer] = MakeMatrix(prev_size, size, 0.0);
                allBiasGradsAcc[layer] = new double[size];

                allPrevWeightGradsAcc[layer] = MakeMatrix(prev_size, size, 0.0);
                allPrevBiasGradsAcc[layer] = new double[size];

                allPrevWeightDeltas[layer] = MakeMatrix(prev_size, size, 0.0);
                allPrevBiasDeltas[layer] = MakeVector(size, 0.01);
            }

            int epoch = 0;
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % 100 == 0 && epoch != maxEpochs)
                {
                    double[] currWts = this.GetWeights();
                    double err = MeanSquaredError(trainData, currWts);
                    Console.WriteLine("epoch = " + epoch + " err = " + err.ToString("F4"));
                }

                // 1. compute and accumulate all gradients
                for (int layer = 1; layer < layer_count; layer++)
                {
                    ZeroOut(allWeightGradsAcc[layer]);// zero-out values from prev iteration
                    ZeroOut(allBiasGradsAcc[layer]);
                }

                double[] xValues = new double[sizes[0]]; // inputs
                double[] tValues = new double[sizes[2]]; // target values
                for (int row = 0; row < trainData.Length; ++row)  // walk thru all training data
                {
                    // no need to visit in random order because all rows processed before any updates ('batch')
                    Array.Copy(trainData[row], xValues, sizes[0]); // get the inputs
                    Array.Copy(trainData[row], sizes[0], tValues, 0, sizes[2]); // get the target values
                    ComputeOutputs(xValues); // copy xValues in, compute outputs using curr weights (and store outputs internally)

                    // compute the h-o gradient term/component as in regular back-prop
                    // this term usually is lower case Greek delta but there are too many other deltas below
                    for (int i = 0; i < sizes[2]; ++i)
                    {
                        double derivative = (1 - this.values[NeuralNetwork.layer_count - 1][i]) * this.values[NeuralNetwork.layer_count - 1][i]; // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                        allGradTerms[2][i] = derivative * (this.values[NeuralNetwork.layer_count - 1][i] - tValues[i]); // careful with O-T vs. T-O, O-T is the most usual
                    }

                    // compute the i-h gradient term/component as in regular back-prop
                    for (int i = 0; i < sizes[1]; ++i)
                    {
                        double derivative = (1 - this.values[1][i]) * (1 + this.values[1][i]); // derivative of tanh = (1 - y) * (1 + y)
                        double sum = 0.0;
                        for (int j = 0; j < sizes[2]; ++j) // each hidden delta is the sum of sizes[2] terms
                        {
                            double x = allGradTerms[2][j] * this.weights[2][i][j];
                            sum += x;
                        }
                        allGradTerms[1][i] = derivative * sum;
                    }

                    for (int layer = layer_count - 1; layer >= 1; layer--)
                    {
                        double[] tst;
                        if (layer == layer_count - 1) { tst = this.values[1]; } else { tst = this.values[0]; }
                        // add input to h-o component to make h-o weight gradients, and accumulate
                        for (int i = 0; i < sizes[layer - 1]; ++i)
                        {
                            for (int j = 0; j < sizes[layer]; ++j)
                            {
                                double grad = allGradTerms[layer][j] * tst[i];
                                allWeightGradsAcc[layer][i][j] += grad;
                            }
                        }

                        // the (hidden-to-) output bias gradients
                        for (int i = 0; i < sizes[layer]; ++i)
                        {
                            double grad = allGradTerms[layer][i] * 1.0; // dummy input
                            allBiasGradsAcc[layer][i] += grad;
                        }
                    }
                } // each row
                  // end compute all gradients

                // update all weights and biases (in any order)
                for (int layer = 1; layer < layer_count; layer++)
                {
                    int size = sizes[layer];
                    int previous_size = sizes[layer - 1];
                    double[][] this_weights = (layer == 1) ? this.weights[1] : this.weights[2];
                    double[] this_biases = (layer == 1) ? this.biases[1] : this.biases[2];

                    // update input-hidden weights
                    for (int i = 0; i < previous_size; ++i)
                    {
                        for (int j = 0; j < size; ++j)
                        {
                            double delta = 0.0;
                            double t = allPrevWeightGradsAcc[layer][i][j] * allWeightGradsAcc[layer][i][j];
                            if (t > 0) // no sign change, increase delta
                            {
                                delta = allPrevWeightDeltas[layer][i][j] * etaPlus; // compute delta
                                if (delta > deltaMax) delta = deltaMax; // keep it in range
                                double tmp = -Math.Sign(allWeightGradsAcc[layer][i][j]) * delta; // determine direction and magnitude
                                this_weights[i][j] += tmp; // update weights
                            }
                            else if (t < 0) // grad changed sign, decrease delta
                            {
                                delta = allPrevWeightDeltas[layer][i][j] * etaMinus; // the delta (not used, but saved for later)
                                if (delta < deltaMin) delta = deltaMin; // keep it in range
                                this_weights[i][j] -= allPrevWeightDeltas[layer][i][j]; // revert to previous weight
                                allWeightGradsAcc[layer][i][j] = 0; // forces next if-then branch, next iteration
                            }
                            else // this happens next iteration after 2nd branch above (just had a change in gradient)
                            {
                                delta = allPrevWeightDeltas[layer][i][j]; // no change to delta
                                                                          // no way should delta be 0 . . .
                                double tmp = -Math.Sign(allWeightGradsAcc[layer][i][j]) * delta; // determine direction
                                this_weights[i][j] += tmp; // update
                            }
                            //Console.WriteLine(allPrevWeightGradsAcc[1][i][j] + " " + allWeightGradsAcc[1][i][j]); Console.ReadLine();

                            allPrevWeightDeltas[layer][i][j] = delta; // save delta
                            allPrevWeightGradsAcc[layer][i][j] = allWeightGradsAcc[layer][i][j]; // save the (accumulated) gradient
                        } // j
                    } // i

                    // update (input-to-) hidden biases
                    for (int i = 0; i < size; ++i)
                    {
                        double delta = 0.0;
                        double t = allPrevBiasGradsAcc[layer][i] * allBiasGradsAcc[layer][i];
                        if (t > 0) // no sign change, increase delta
                        {
                            delta = allPrevBiasDeltas[layer][i] * etaPlus; // compute delta
                            if (delta > deltaMax) delta = deltaMax;
                            double tmp = -Math.Sign(allBiasGradsAcc[layer][i]) * delta; // determine direction
                            this_biases[i] += tmp; // update
                        }
                        else if (t < 0) // grad changed sign, decrease delta
                        {
                            delta = allPrevBiasDeltas[layer][i] * etaMinus; // the delta (not used, but saved later)
                            if (delta < deltaMin) delta = deltaMin;
                            this_biases[i] -= allPrevBiasDeltas[layer][i]; // revert to previous weight
                            allBiasGradsAcc[layer][i] = 0; // forces next branch, next iteration
                        }
                        else // this happens next iteration after 2nd branch above (just had a change in gradient)
                        {
                            delta = allPrevBiasDeltas[layer][i]; // no change to delta

                            if (delta > deltaMax) delta = deltaMax;
                            else if (delta < deltaMin) delta = deltaMin;
                            // no way should delta be 0 . . .
                            double tmp = -Math.Sign(allBiasGradsAcc[layer][i]) * delta; // determine direction
                            this_biases[i] += tmp; // update
                        }
                        allPrevBiasDeltas[layer][i] = delta;
                        allPrevBiasGradsAcc[layer][i] = allBiasGradsAcc[layer][i];
                    }
                }// layer
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
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (this.sizes[0] * this.sizes[1]) +
                (this.sizes[1] * this.sizes[NeuralNetwork.layer_count - 1]) + this.sizes[1] +
                this.sizes[NeuralNetwork.layer_count - 1];
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0; // points into weights param

            for (int layer = 1; layer < NeuralNetwork.layer_count; layer++)
            {
                for (int i = 0; i < this.sizes[layer - 1]; ++i)
                    for (int j = 0; j < this.sizes[layer]; ++j)
                        this.weights[layer][i][j] = weights[k++];
                for (int i = 0; i < this.sizes[layer]; ++i)
                    this.biases[layer][i] = weights[k++];
            }
        }

        public double[] GetWeights()
        {
            int numWeights = (this.sizes[0] * this.sizes[1]) +
                (this.sizes[1] * this.sizes[NeuralNetwork.layer_count - 1]) + this.sizes[1] +
                this.sizes[NeuralNetwork.layer_count - 1];
            double[] result = new double[numWeights];
            int k = 0;
            for (int layer = 1; layer < NeuralNetwork.layer_count; layer++)
            {
                for (int i = 0; i < this.weights[layer].Length; ++i)
                    for (int j = 0; j < this.weights[layer][0].Length; ++j)
                        result[k++] = this.weights[layer][i][j];
                for (int i = 0; i < this.biases[layer].Length; ++i)
                    result[k++] = this.biases[layer][i];
            }
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            Array.Copy(xValues, this.values[0], this.sizes[0]);
            for (int layer = 1; layer < NeuralNetwork.layer_count; layer++)
            {
                double[] sums = new double[this.sizes[layer]]; // hidden nodes sums scratch array
                Array.Copy(this.biases[layer], sums, this.sizes[layer]);
                for (int j = 0; j < this.sizes[layer]; ++j)  // compute i-h sum of weights * inputs
                    for (int i = 0; i < this.sizes[layer - 1]; ++i)
                        sums[j] += this.values[layer - 1][i] * this.weights[layer][i][j]; // note +=
                if (layer < NeuralNetwork.layer_count - 1)
                {
                    for (int i = 0; i < this.sizes[layer]; ++i)   // apply activation
                        this.values[layer][i] = HyperTan(sums[i]); // hard-coded
                }
                else
                {
                    this.values[NeuralNetwork.layer_count - 1] = Softmax(sums);
                }
            }

            double[] retResult = new double[this.sizes[NeuralNetwork.layer_count - 1]]; // could define a GetOutputs method instead
            Array.Copy(this.values[NeuralNetwork.layer_count - 1], retResult, retResult.Length);
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
            this.SetWeights(weights);
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[this.sizes[0]]; // inputs
            double[] tValues = new double[this.sizes[NeuralNetwork.layer_count - 1]]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, this.sizes[0]); // parse data into x-values and t-values
                Array.Copy(testData[i], this.sizes[0], tValues, 0, this.sizes[NeuralNetwork.layer_count - 1]);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y, double epsilon)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
        }

        public double MeanSquaredError(double[][] trainData, double[] weights)
        {
            this.SetWeights(weights); // copy the weights to evaluate in

            double[] xValues = new double[this.sizes[0]]; // this.values[0]
            double[] tValues = new double[this.sizes[NeuralNetwork.layer_count - 1]]; // targets
            double sumSquaredError = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // walk through each training data item
            {
                // following assumes data has all x-values first, followed by y-values!
                Array.Copy(trainData[i], xValues, this.sizes[0]); // extract inputs
                Array.Copy(trainData[i], this.sizes[0], tValues, 0, this.sizes[NeuralNetwork.layer_count - 1]); // extract targets
                double[] yValues = this.ComputeOutputs(xValues);
                for (int j = 0; j < yValues.Length; ++j)
                    sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
            }
            return sumSquaredError / trainData.Length;
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

    } // NeuralNetwork
}
