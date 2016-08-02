using System;

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
            int seed = 0;

            Console.WriteLine("\nGenerating " + numRows +
              " artificial data items with " + numInput + " features");
            double[][] allData = MakeAllData(numInput, numHidden, numOutput,
              numRows, seed);
            Console.WriteLine("Done");

            Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, 0.80, seed, out trainData, out testData);
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
            Console.WriteLine("\nAccuracy on training data = " +
              trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData, weights);
            Console.WriteLine("\nAccuracy on test data = " +
              testAcc.ToString("F4"));

            Console.WriteLine("\nEnd neural network with Resilient Propagation demo\n");
            Console.ReadLine();
        } // Main

        static double[][] MakeAllData(int numInput, int numHidden, int numOutput,
          int numRows, int seed)
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

        static void MakeTrainTest(double[][] allData, double trainPct, int seed,
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
        private int numInput; // number input nodes
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights; // input-hidden
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights; // hidden-output
        private double[] oBiases;
        private double[] outputs;

        private Random rnd;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];

            this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

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
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.SetWeights(initialWeights);
        }

        public double[] TrainRPROP(double[][] trainData, int maxEpochs) // using RPROP
        {
            // there is an accumulated gradient and a previous gradient for each weight and bias
            double[] hGradTerms = new double[numHidden]; // intermediate val for h-o weight gradients
            double[] oGradTerms = new double[numOutput]; // output gradients

            double[][] hoWeightGradsAcc = MakeMatrix(numHidden, numOutput, 0.0); // accumulated over all training data
            double[][] ihWeightGradsAcc = MakeMatrix(numInput, numHidden, 0.0);
            double[] oBiasGradsAcc = new double[numOutput];
            double[] hBiasGradsAcc = new double[numHidden];

            double[][] hoPrevWeightGradsAcc = MakeMatrix(numHidden, numOutput, 0.0); // accumulated, previous iteration
            double[][] ihPrevWeightGradsAcc = MakeMatrix(numInput, numHidden, 0.0);
            double[] oPrevBiasGradsAcc = new double[numOutput];
            double[] hPrevBiasGradsAcc = new double[numHidden];

            // must save previous weight deltas
            double[][] hoPrevWeightDeltas = MakeMatrix(numHidden, numOutput, 0.01);
            double[][] ihPrevWeightDeltas = MakeMatrix(numInput, numHidden, 0.01);
            double[] oPrevBiasDeltas = MakeVector(numOutput, 0.01);
            double[] hPrevBiasDeltas = MakeVector(numHidden, 0.01);

            double etaPlus = 1.2; // values are from the paper
            double etaMinus = 0.5;
            double deltaMax = 50.0;
            double deltaMin = 1.0E-6;

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
                ZeroOut(hoWeightGradsAcc); // zero-out values from prev iteration
                ZeroOut(ihWeightGradsAcc);
                ZeroOut(oBiasGradsAcc);
                ZeroOut(hBiasGradsAcc);

                double[] xValues = new double[numInput]; // inputs
                double[] tValues = new double[numOutput]; // target values
                for (int row = 0; row < trainData.Length; ++row)  // walk thru all training data
                {
                    // no need to visit in random order because all rows processed before any updates ('batch')
                    Array.Copy(trainData[row], xValues, numInput); // get the inputs
                    Array.Copy(trainData[row], numInput, tValues, 0, numOutput); // get the target values
                    ComputeOutputs(xValues); // copy xValues in, compute outputs using curr weights (and store outputs internally)

                    // compute the h-o gradient term/component as in regular back-prop
                    // this term usually is lower case Greek delta but there are too many other deltas below
                    for (int i = 0; i < numOutput; ++i)
                    {
                        double derivative = (1 - outputs[i]) * outputs[i]; // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                        oGradTerms[i] = derivative * (outputs[i] - tValues[i]); // careful with O-T vs. T-O, O-T is the most usual
                    }

                    // compute the i-h gradient term/component as in regular back-prop
                    for (int i = 0; i < numHidden; ++i)
                    {
                        double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // derivative of tanh = (1 - y) * (1 + y)
                        double sum = 0.0;
                        for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
                        {
                            double x = oGradTerms[j] * hoWeights[i][j];
                            sum += x;
                        }
                        hGradTerms[i] = derivative * sum;
                    }

                    // add input to h-o component to make h-o weight gradients, and accumulate
                    for (int i = 0; i < numHidden; ++i)
                    {
                        for (int j = 0; j < numOutput; ++j)
                        {
                            double grad = oGradTerms[j] * hOutputs[i];
                            hoWeightGradsAcc[i][j] += grad;
                        }
                    }

                    // the (hidden-to-) output bias gradients
                    for (int i = 0; i < numOutput; ++i)
                    {
                        double grad = oGradTerms[i] * 1.0; // dummy input
                        oBiasGradsAcc[i] += grad;
                    }

                    // add input term to i-h component to make i-h weight gradients and accumulate
                    for (int i = 0; i < numInput; ++i)
                    {
                        for (int j = 0; j < numHidden; ++j)
                        {
                            double grad = hGradTerms[j] * inputs[i];
                            ihWeightGradsAcc[i][j] += grad;
                        }
                    }

                    // the (input-to-) hidden bias gradient
                    for (int i = 0; i < numHidden; ++i)
                    {
                        double grad = hGradTerms[i] * 1.0;
                        hBiasGradsAcc[i] += grad;
                    }
                } // each row
                  // end compute all gradients

                // update all weights and biases (in any order)

                // update input-hidden weights
                double delta = 0.0;

                for (int i = 0; i < numInput; ++i)
                {
                    for (int j = 0; j < numHidden; ++j)
                    {
                        if (ihPrevWeightGradsAcc[i][j] * ihWeightGradsAcc[i][j] > 0) // no sign change, increase delta
                        {
                            delta = ihPrevWeightDeltas[i][j] * etaPlus; // compute delta
                            if (delta > deltaMax) delta = deltaMax; // keep it in range
                            double tmp = -Math.Sign(ihWeightGradsAcc[i][j]) * delta; // determine direction and magnitude
                            ihWeights[i][j] += tmp; // update weights
                        }
                        else if (ihPrevWeightGradsAcc[i][j] * ihWeightGradsAcc[i][j] < 0) // grad changed sign, decrease delta
                        {
                            delta = ihPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved for later)
                            if (delta < deltaMin) delta = deltaMin; // keep it in range
                            ihWeights[i][j] -= ihPrevWeightDeltas[i][j]; // revert to previous weight
                            ihWeightGradsAcc[i][j] = 0; // forces next if-then branch, next iteration
                        }
                        else // this happens next iteration after 2nd branch above (just had a change in gradient)
                        {
                            delta = ihPrevWeightDeltas[i][j]; // no change to delta
                                                              // no way should delta be 0 . . . 
                            double tmp = -Math.Sign(ihWeightGradsAcc[i][j]) * delta; // determine direction
                            ihWeights[i][j] += tmp; // update
                        }
                        //Console.WriteLine(ihPrevWeightGradsAcc[i][j] + " " + ihWeightGradsAcc[i][j]); Console.ReadLine();

                        ihPrevWeightDeltas[i][j] = delta; // save delta
                        ihPrevWeightGradsAcc[i][j] = ihWeightGradsAcc[i][j]; // save the (accumulated) gradient
                    } // j
                } // i

                // update (input-to-) hidden biases
                for (int i = 0; i < numHidden; ++i)
                {
                    if (hPrevBiasGradsAcc[i] * hBiasGradsAcc[i] > 0) // no sign change, increase delta
                    {
                        delta = hPrevBiasDeltas[i] * etaPlus; // compute delta
                        if (delta > deltaMax) delta = deltaMax;
                        double tmp = -Math.Sign(hBiasGradsAcc[i]) * delta; // determine direction
                        hBiases[i] += tmp; // update
                    }
                    else if (hPrevBiasGradsAcc[i] * hBiasGradsAcc[i] < 0) // grad changed sign, decrease delta
                    {
                        delta = hPrevBiasDeltas[i] * etaMinus; // the delta (not used, but saved later)
                        if (delta < deltaMin) delta = deltaMin;
                        hBiases[i] -= hPrevBiasDeltas[i]; // revert to previous weight
                        hBiasGradsAcc[i] = 0; // forces next branch, next iteration
                    }
                    else // this happens next iteration after 2nd branch above (just had a change in gradient)
                    {
                        delta = hPrevBiasDeltas[i]; // no change to delta

                        if (delta > deltaMax) delta = deltaMax;
                        else if (delta < deltaMin) delta = deltaMin;
                        // no way should delta be 0 . . . 
                        double tmp = -Math.Sign(hBiasGradsAcc[i]) * delta; // determine direction
                        hBiases[i] += tmp; // update
                    }
                    hPrevBiasDeltas[i] = delta;
                    hPrevBiasGradsAcc[i] = hBiasGradsAcc[i];
                }

                // update hidden-to-output weights
                for (int i = 0; i < numHidden; ++i)
                {
                    for (int j = 0; j < numOutput; ++j)
                    {
                        if (hoPrevWeightGradsAcc[i][j] * hoWeightGradsAcc[i][j] > 0) // no sign change, increase delta
                        {
                            delta = hoPrevWeightDeltas[i][j] * etaPlus; // compute delta
                            if (delta > deltaMax) delta = deltaMax;
                            double tmp = -Math.Sign(hoWeightGradsAcc[i][j]) * delta; // determine direction
                            hoWeights[i][j] += tmp; // update
                        }
                        else if (hoPrevWeightGradsAcc[i][j] * hoWeightGradsAcc[i][j] < 0) // grad changed sign, decrease delta
                        {
                            delta = hoPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved later)
                            if (delta < deltaMin) delta = deltaMin;
                            hoWeights[i][j] -= hoPrevWeightDeltas[i][j]; // revert to previous weight
                            hoWeightGradsAcc[i][j] = 0; // forces next branch, next iteration
                        }
                        else // this happens next iteration after 2nd branch above (just had a change in gradient)
                        {
                            delta = hoPrevWeightDeltas[i][j]; // no change to delta
                                                              // no way should delta be 0 . . . 
                            double tmp = -Math.Sign(hoWeightGradsAcc[i][j]) * delta; // determine direction
                            hoWeights[i][j] += tmp; // update
                        }
                        hoPrevWeightDeltas[i][j] = delta; // save delta
                        hoPrevWeightGradsAcc[i][j] = hoWeightGradsAcc[i][j]; // save the (accumulated) gradients
                    } // j
                } // i

                // update (hidden-to-) output biases
                for (int i = 0; i < numOutput; ++i)
                {
                    if (oPrevBiasGradsAcc[i] * oBiasGradsAcc[i] > 0) // no sign change, increase delta
                    {
                        delta = oPrevBiasDeltas[i] * etaPlus; // compute delta
                        if (delta > deltaMax) delta = deltaMax;
                        double tmp = -Math.Sign(oBiasGradsAcc[i]) * delta; // determine direction
                        oBiases[i] += tmp; // update
                    }
                    else if (oPrevBiasGradsAcc[i] * oBiasGradsAcc[i] < 0) // grad changed sign, decrease delta
                    {
                        delta = oPrevBiasDeltas[i] * etaMinus; // the delta (not used, but saved later)
                        if (delta < deltaMin) delta = deltaMin;
                        oBiases[i] -= oPrevBiasDeltas[i]; // revert to previous weight
                        oBiasGradsAcc[i] = 0; // forces next branch, next iteration
                    }
                    else // this happens next iteration after 2nd branch above (just had a change in gradient)
                    {
                        delta = oPrevBiasDeltas[i]; // no change to delta
                                                    // no way should delta be 0 . . . 
                        double tmp = -Math.Sign(hBiasGradsAcc[i]) * delta; // determine direction
                        oBiases[i] += tmp; // update
                    }
                    oPrevBiasDeltas[i] = delta;
                    oPrevBiasGradsAcc[i] = oBiasGradsAcc[i];
                }
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
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];
            // note: no need to copy x-values unless you implement a ToString and want to see them.
            // more efficient is to simply use the xValues[] directly.

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

            for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this.hBiases[i];

            for (int i = 0; i < numHidden; ++i)   // apply activation
                this.hOutputs[i] = HyperTan(hSums[i]); // hard-coded

            for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
            Array.Copy(this.outputs, retResult, retResult.Length);
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
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // parse data into x-values and t-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
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

            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double sumSquaredError = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // walk through each training data item
            {
                // following assumes data has all x-values first, followed by y-values!
                Array.Copy(trainData[i], xValues, numInput); // extract inputs
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // extract targets
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

} // ns
