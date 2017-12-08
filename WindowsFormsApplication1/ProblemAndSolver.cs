using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Diagnostics;
using System.Runtime.Serialization;

namespace TSP
{

    class ProblemAndSolver
    {
        private class TSPSolution
        {
            /// <summary>
            /// we use the representation [cityB,cityA,cityC] 
            /// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
            /// and the edge from cityC to cityB is the final edge in the path.  
            /// You are, of course, free to use a different representation if it would be more convenient or efficient 
            /// for your data structure(s) and search algorithm. 
            /// </summary>
            public ArrayList
                Route;

            public double Bound { get; set; }

            /// <summary>
            /// constructor
            /// </summary>
            /// <param name="iroute">a (hopefully) valid tour</param>
            public TSPSolution(ArrayList iroute)
            {
                Route = new ArrayList(iroute);
            }

            /// <summary>
            /// Compute the cost of the current route.  
            /// Note: This does not check that the route is complete.
            /// It assumes that the route passes from the last city back to the first city. 
            /// </summary>
            /// <returns></returns>
            public double costOfRoute()
            {
                // go through each edge in the route and add up the cost. 
                int x;
                City here;
                double cost = 0D;

                for (x = 0; x < Route.Count - 1; x++)
                {
                    here = Route[x] as City;
                    cost += here.costToGetTo(Route[x + 1] as City);
                }

                // go from the last city to the first. 
                here = Route[Route.Count - 1] as City;
                cost += here.costToGetTo(Route[0] as City);
                return cost;
            }
        }

        #region Private members 

        /// <summary>
        /// Default number of cities (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Problem Size text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int DEFAULT_SIZE = 25;

        /// <summary>
        /// Default time limit (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Time text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int TIME_LIMIT = 60;        //in seconds

        private const int CITY_ICON_SIZE = 5;


        // For normal and hard modes:
        // hard mode only
        private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

        /// <summary>
        /// the cities in the current problem.
        /// </summary>
        private City[] Cities;
        /// <summary>
        /// a route through the current problem, useful as a temporary variable. 
        /// </summary>
        private ArrayList Route;
        /// <summary>
        /// best solution so far. 
        /// </summary>
        private TSPSolution bssf;

        /// <summary>
        /// how to color various things. 
        /// </summary>
        private Brush cityBrushStartStyle;
        private Brush cityBrushStyle;
        private Pen routePenStyle;


        /// <summary>
        /// keep track of the seed value so that the same sequence of problems can be 
        /// regenerated next time the generator is run. 
        /// </summary>
        private int _seed;
        /// <summary>
        /// number of cities to include in a problem. 
        /// </summary>
        private int _size;

        /// <summary>
        /// Difficulty level
        /// </summary>
        private HardMode.Modes _mode;

        /// <summary>
        /// random number generator. 
        /// </summary>
        private Random rnd;

        /// <summary>
        /// time limit in milliseconds for state space search
        /// can be used by any solver method to truncate the search and return the BSSF
        /// </summary>
        private int time_limit;
        #endregion

        #region Public members

        /// <summary>
        /// These three constants are used for convenience/clarity in populating and accessing the results array that is passed back to the calling Form
        /// </summary>
        public const int COST = 0;           
        public const int TIME = 1;
        public const int COUNT = 2;
        
        public int Size
        {
            get { return _size; }
        }

        public int Seed
        {
            get { return _seed; }
        }
        #endregion

        #region Constructors
        public ProblemAndSolver()
        {
            this._seed = 1; 
            rnd = new Random(1);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }

        public ProblemAndSolver(int seed)
        {
            this._seed = seed;
            rnd = new Random(seed);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }
        #endregion

        #region Private Methods

        /// <summary>
        /// Reset the problem instance.
        /// </summary>
        private void resetData()
        {

            Cities = new City[_size];
            Route = new ArrayList(_size);
            bssf = null;

            if (_mode == HardMode.Modes.Easy)
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
            }
            else // Medium and hard
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
            }

            HardMode mm = new HardMode(this._mode, this.rnd, Cities);
            if (_mode == HardMode.Modes.Hard)
            {
                int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
                mm.removePaths(edgesToRemove);
            }
            City.setModeManager(mm);

            cityBrushStyle = new SolidBrush(Color.Black);
            cityBrushStartStyle = new SolidBrush(Color.Red);
            routePenStyle = new Pen(Color.Blue,1);
            routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode)
        {
            this._size = size;
            this._mode = mode;
            resetData();
        }

        /// <summary>
        /// make a new problem with the given size, now including timelimit paremeter that was added to form.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode, int timelimit)
        {
            this._size = size;
            this._mode = mode;
            this.time_limit = timelimit*1000;                                   //convert seconds to milliseconds
            resetData();
        }

        /// <summary>
        /// return a copy of the cities in this problem. 
        /// </summary>
        /// <returns>array of cities</returns>
        public City[] GetCities()
        {
            City[] retCities = new City[Cities.Length];
            Array.Copy(Cities, retCities, Cities.Length);
            return retCities;
        }

        /// <summary>
        /// draw the cities in the problem.  if the bssf member is defined, then
        /// draw that too. 
        /// </summary>
        /// <param name="g">where to draw the stuff</param>
        public void Draw(Graphics g)
        {
            float width  = g.VisibleClipBounds.Width-45F;
            float height = g.VisibleClipBounds.Height-45F;
            Font labelFont = new Font("Arial", 10);

            // Draw lines
            if (bssf != null)
            {
                // make a list of points. 
                Point[] ps = new Point[bssf.Route.Count];
                int index = 0;
                foreach (City c in bssf.Route)
                {
                    if (index < bssf.Route.Count -1)
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[index+1]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    else 
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[0]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
                }

                if (ps.Length > 0)
                {
                    g.DrawLines(routePenStyle, ps);
                    g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
                }

                // draw the last line. 
                g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
            }

            // Draw city dots
            foreach (City c in Cities)
            {
                g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
            }

        }

        /// <summary>
        ///  return the cost of the best solution so far. 
        /// </summary>
        /// <returns></returns>
        public double costOfBssf ()
        {
            if (bssf != null)
                return (bssf.costOfRoute());
            else
                return -1D; 
        }

        /// <summary>
        /// This is the entry point for the default solver
        /// which just finds a valid random tour 
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] defaultSolveProblem()
        {
            int i, swap, temp, count=0;
            string[] results = new string[3];
            int[] perm = new int[Cities.Length];
            Route = new ArrayList();
            Random rnd = new Random();
            Stopwatch timer = new Stopwatch();

            timer.Start();

            do
            {
                for (i = 0; i < perm.Length; i++)                                 // create a random permutation template
                    perm[i] = i;
                for (i = 0; i < perm.Length; i++)
                {
                    swap = i;
                    while (swap == i)
                        swap = rnd.Next(0, Cities.Length);
                    temp = perm[i];
                    perm[i] = perm[swap];
                    perm[swap] = temp;
                }
                Route.Clear();
                for (i = 0; i < Cities.Length; i++)                            // Now build the route using the random permutation 
                {
                    Route.Add(Cities[perm[i]]);
                }
                bssf = new TSPSolution(Route);
                count++;
            } while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
            timer.Stop();

            results[COST] = costOfBssf().ToString();                          // load results array
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }

        /// <summary>
        /// performs a Branch and Bound search of the state space of partial tours
        /// stops when time limit expires and uses BSSF as solution
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] bBSolveProblem()
        {
            string[] results = new string[3];
            Stopwatch timer = new Stopwatch();
            State childState,
                currentState;
            TSPSolution possibleSolution;
            double[][] edges;
            double cost;
            int cityIndex,
                count = 0,
                numLeaves = 0,
                numPruned = 0,
                numStatesCreated = 0,
                size = Cities.Length;

            // Run Branch & Bound
            timer.Start();

            // Run the greedy solver to get a baseline best solution so far
            GreedySolver gs = new GreedySolver(Cities);
            bssf = gs.Solution();

            // Run branch and bound to find the optimal solution (if we can)
            // Create the start state and add the queue
            currentState = State.CreateState(Cities, 0); numStatesCreated++;
            currentState.Reduce();

            BinaryHeapQueue queue = new BinaryHeapQueue(Cities.Length);
            queue.Insert(currentState);

            // Repeat while there are items in the queue
            // At worst this will loop for every possible tree node, so 2^n times.
            // This gives a worst case run time of O(n^2 * 2^n)
            while (!queue.Empty())
            {
                currentState = queue.DeleteMin();
                if (currentState.Bound() < bssf.Bound)
                {
                    // Create the child nodes
                    // For every valid edge out of this state, create a child state
                    edges = currentState.Edges();
                    cityIndex = currentState.CityId();

                    for (int to = 0; to < size; to++)// Loops n times
                    {
                        // Create a child state for each valid edge out of this node
                        // In addition if there is a edge, do a quick check to see the cost of the edge is out of the bound before creating
                        if (edges[cityIndex][to] != double.PositiveInfinity && (edges[cityIndex][to] + currentState.Bound()) < bssf.Bound)
                        {
                            childState = currentState.CreateChildState(Cities[to], cityIndex, to); numStatesCreated++;
                            childState.Reduce();
                            if (childState.Depth() == size)
                            {
                                // Leaf node
                                numLeaves++;

                                // Check for a valid solution
                                possibleSolution = new TSPSolution(childState.Tour());
                                cost = possibleSolution.costOfRoute();
                                if (cost != double.PositiveInfinity && cost < bssf.Bound)
                                {
                                    // Set BSSF
                                    bssf = possibleSolution;
                                    bssf.Bound = cost;
                                    count++;
                                }
                            }
                            else if (childState.Bound() < bssf.Bound)
                            {
                                queue.Insert(childState);
                            }
                            else
                            {
                                numPruned++;
                            }
                        }
                        // If the timer has expired, stop
                        if (timer.ElapsedMilliseconds >= time_limit)
                        {
                            queue.MarkAsEmpty();
                            break;
                        }
                    }
                }
                else
                {
                    // State from the queue has a larger bound than the bssf, prune this state
                    numPruned++;
                }
            }
            timer.Stop();


            // Prepare and return the results
            results[COST] = bssf.costOfRoute().ToString();
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            // Report
            Debug.WriteLine("Pruned: " + numPruned + " Max Queue Size: " + queue.MaxSize());
            Debug.WriteLine("Created: " + numStatesCreated + " Leaves: " + numLeaves);

            return results;
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        // These additional solver methods will be implemented as part of the group project.
        ////////////////////////////////////////////////////////////////////////////////////////////

        /// <summary>
        /// finds the greedy tour starting from each city and keeps the best (valid) one
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
         public string[] greedySolveProblem()
        {
            string[] results = new string[3];
            Stopwatch timer = new Stopwatch();
            timer.Start();
            //Obtain a BSSF by using a GREEDY approach.
            List<ArrayList> routes = GenerateGreddyRoutes();
            bssf = new TSPSolution(BestRoute(routes));

            timer.Stop();
            results[COST] = bssf.costOfRoute().ToString();
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = "";

            return results;
        }

        public List<ArrayList> GenerateGreddyRoutes()
        {
            List<ArrayList> routes = new List<ArrayList>();
            int cityCount = Cities.Length;
            bool routeCreated = false;

            for(int i = 0; i < cityCount; i++)
            {
                //For each city
                ArrayList route = new ArrayList();
                routeCreated = GenerateRoute(route, i);
                if (routeCreated)
                {
                    routes.Add(route);
                }
            }  
            return routes; 
        }

        public bool GenerateRoute(ArrayList route, int startCity)
        {
            List<int> remainingCities = new List<int>();
            City currentCity = Cities[startCity];
            route.Add(currentCity);

            //Make a queue to help keep track of remaining cities
            //Remember to not add city 0
            int cityCount = Cities.Length;
            for (int i = 0; i < cityCount; i++)
            {
                if (i != startCity)
                {
                    remainingCities.Add(i);
                }
            }

            //Initialixe helper variables
            City neighborCity = null;
            int neighborCityIndex = 0;
            double neighborCost = Double.PositiveInfinity;
            double cost = 0;

            while (remainingCities.Count != 0)
            {
                //For each remaining city
                int size = remainingCities.Count;
                for (int i = 0; i < size; i++)
                {
                    int remainingCity = remainingCities[i];
                    cost = currentCity.costToGetTo(Cities[remainingCity]);
                    if (cost < neighborCost)
                    {
                        neighborCity = Cities[remainingCity];
                        neighborCost = cost;
                        neighborCityIndex = remainingCity;
                    }
                }

                //If the cost from the one city to the next comes out to be infinite then this route failed, and we have to start over.
                if (cost == Double.PositiveInfinity)
                {
                    return false;
                }

                //As it ends, we will have the closest neighbor to the current city
                route.Add(neighborCity);
                remainingCities.Remove(neighborCityIndex);
                currentCity = neighborCity;
                neighborCost = Double.PositiveInfinity;
            }

            //Verify that the last city in the route can make it back to the first
            City lastCity = (City)route[route.Count - 1];
            City beginCity = Cities[startCity];
            if (lastCity.costToGetTo(beginCity) == Double.PositiveInfinity)
            {
                return false;
            }

            //If we make it through, we found a route
            return true;
        }

        public ArrayList BestRoute(List<ArrayList> routes)
        {
            int routeCount = routes.Count;
            double bestCostSoFar = Double.PositiveInfinity;
            int BCSFIndex = 0;

            for (int i = 0; i < routeCount; i++)
            {
                TSPSolution solution = new TSPSolution(routes[i]);
                double currentCost = solution.costOfRoute();
                if(currentCost < bestCostSoFar)
                {
                    bestCostSoFar = currentCost;
                    BCSFIndex = i;
                }
            }
            return routes[BCSFIndex];
        }


        public string[] fancySolveProblem()
        {
            greedySolveProblem();
            string[] results = new string[3];
            Stopwatch timer = new Stopwatch();
            int solutionCount = 0;
            timer.Start();

            City[] newCities = new City[Cities.Length];
            for (int i = 0; i < Cities.Length; i++)
            {
                newCities[i] = bssf.Route[i] as City;
            }
            double minChange, change;
            do
            {
                if (timer.ElapsedMilliseconds >= time_limit)
                {
                    break;
                }
                minChange = 0;
                int iMin = 0;
                int kMin = 0;
                for (int i = 0; i < newCities.Length - 2; i++)
                {
                    for (int k = i + 2; k < newCities.Length; k++)
                    {
                        if (k == newCities.Length - 1)
                        {
                            // Check the edge from the last city in the tour to the first
                            change = newCities[i].costToGetTo(newCities[k]) + newCities[i + 1].costToGetTo(newCities[0]) -
                                newCities[i].costToGetTo(newCities[i + 1]) - newCities[k].costToGetTo(newCities[0]);
                        }
                        else
                        {
                            change = newCities[i].costToGetTo(newCities[k]) + newCities[i + 1].costToGetTo(newCities[k + 1]) -
                                newCities[i].costToGetTo(newCities[i + 1]) - newCities[k].costToGetTo(newCities[k + 1]);
                        }
                        if (change < minChange)
                        {
                            minChange = change;
                            iMin = i;
                            kMin = k;
                        }
                    }
                }
                if (minChange < 0)
                {
                    // Swap
                    City[] newRoute = new City[Cities.Length];
                    for (int i = 0; i <= iMin; i++)
                    {
                        newRoute[i] = newCities[i];
                    }
                    int k = kMin;
                    for (int i = iMin + 1; i <= kMin; i++)
                    {
                        newRoute[i] = newCities[k];
                        k--;
                    }
                    for (int i = kMin + 1; i < newRoute.Length; i++)
                    {
                        newRoute[i] = newCities[i];
                    }
                    solutionCount++;
                    Route = new ArrayList(newRoute);
                    bssf = new TSPSolution(Route);
                    newCities = newRoute;
                }
            } while (minChange < 0);



            timer.Stop();

            results[COST] = costOfBssf().ToString();
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = solutionCount.ToString();

            return results;
        }
        #endregion

        /**
         * GreedSolver class.
         * Takes O(n^2) space.
         */
        private class GreedySolver
        {
            const int NO_NODE = -1;

            private ArrayList route;
            private City[] cities;
            private TSPSolution bssf;

            private bool solutionFound = false;
            private bool[] visited;
            private int size;
            private int backtrackingCount = 0;
            private int[] backtrackingSize;
            private int[][] backtracking;

            /**
             * Create a new greedy solver.
             * Runs in O(n) and takes O(n^2) space.
             */
            public GreedySolver(City[] cities)
            {
                this.cities = cities;
                size = cities.Length;
                route = new ArrayList(size);
                visited = new bool[size];

                // Initialize the structures used for backtracking
                // O(n)
                backtrackingSize = new int[size];
                backtracking = new int[size][];
                for (int i = 0; i < size; i++)
                {
                    backtracking[i] = new int[size];
                }
            }

            /**
             * Find a solution using a greedy algorithm.
             * This works by doing a depth first search taking the least expensive edge out of every city
             * until a valid solution is found.
             * Runs in O(n^2)
             */
            public TSPSolution Solution()
            {
                for (int i = 0; i < size; i++) // O(n)
                {
                    // Try to find a solution starting from this node
                    GreedyExplore(i);

                    if (solutionFound)
                    {
                        // Found a solution, stop exploring
                        break;
                    }
                    else
                    {
                        // No solution found by starting at this node, reset backtracking
                        for (int j = 0; j < size; j++) // O(n)
                        {
                            backtrackingSize[j] = 0;
                        }
                    }
                }

                return bssf;
            }

            public int BacktrackCount()
            {
                return backtrackingCount;
            }

            /**
             * Greedily explores the child nodes of the given node.
             * Runs in O(n^2)
             */
            private void GreedyExplore(int nodeIndex)
            {
                // Explore this node //
                double min,
                       cost;
                int minIndex;

                // Mark this node as visited
                visited[nodeIndex] = true;
                route.Add(cities[nodeIndex]);

                // Check if we have a solution
                if (route.Count == size)
                {
                    bssf = new TSPSolution(route);
                    bssf.Bound = bssf.costOfRoute();
                    if (bssf.Bound != double.PositiveInfinity)
                    {
                        // Found a solution
                        solutionFound = true;
                        return;
                    }
                    // Route is not a solution, we will need to backtrack
                    backtrackingCount++;
                }

                do
                {
                    // Find the smallest route out of this node
                    minIndex = NO_NODE;
                    min = double.PositiveInfinity;

                    for (int i = 0; i < size; i++) // O(n)
                    {
                        // Make sure that the node i is valid
                        // To be valid it can't be the same node that is being explored, it can't already be in the route (visited), and
                        // we need to have not already tried checking this node (backtracking).
                        if (i != nodeIndex && !visited[i] && !AlreadyChecked(nodeIndex, i))// O(n)
                        {
                            cost = cities[i].costToGetTo(cities[nodeIndex]);
                            if (cost != double.PositiveInfinity && cost < min)
                            {
                                min = cost;
                                minIndex = i;
                            }
                        }
                    }
                    if (minIndex != NO_NODE)
                    {
                        // Mark that we took this search path for this node
                        backtracking[nodeIndex][backtrackingSize[nodeIndex]] = minIndex;
                        backtrackingSize[nodeIndex]++;

                        // Explore our child node
                        GreedyExplore(minIndex);
                    }
                    // Repeat until a solution is found or there are no more children. Repeats 0 to n times.
                } while (minIndex != NO_NODE && !solutionFound);

                if (!solutionFound)
                {
                    // If we've gotten to this point without finding a solution that means we have explored every valid route out of this node
                    // and none of them resulted in a valid tour, so we will need to do some backtracking and try a different path.
                    // Clear this node to be visited again
                    route.RemoveAt(route.Count - 1);
                    visited[nodeIndex] = false;
                }
            }

            /**
             * Indicates if the node has already been checked for a solution.
             * Runs in O(n) time at worst.
             */
            private bool AlreadyChecked(int nodeIndex, int i)
            {
                for (int j = 0; j < backtrackingSize[nodeIndex]; j++)
                {
                    if (backtracking[nodeIndex][j] == i)
                    {
                        return true;
                    }
                }
                return false;
            }
        }

        /**
         * Holds a search state in the branch and bound algorithm.
         * Takes O(n^2) space.
         */
        private class State
        {
            private double bound;
            private double[][] edges;// O(n^2) space
            private City[] tour;// O(n) space
            private int cityId;

            /**
             * Create a new state.
             * Runs in O(1) and takes O(n^2) space.
             */
            public State(double bound, double[][] edges, City[] tour, int cityId)
            {
                this.bound = bound;
                this.edges = edges;
                this.tour = tour;
                this.cityId = cityId;
            }

            /**
             * Create a new state from the given cities array.
             * Runs in O(n^2)
             */
            public static State CreateState(City[] cities, int startCityId)
            {
                // Create a state from the provided list of cities
                // Row = from
                // Col = to
                int size = cities.Length;
                double[][] edges = new double[size][];
                for (int from = 0; from < size; from++)// Runs in O(n^2) (this repeats n times, and has an inner loop that repeats n times)
                {
                    edges[from] = new double[size];
                    for (int to = 0; to < size; to++)// O(n)
                    {
                        if (to == from)
                        {
                            edges[from][to] = double.PositiveInfinity;
                        }
                        else
                        {
                            edges[from][to] = cities[from].costToGetTo(cities[to]);
                        }
                    }
                }

                City[] tour = new City[1];
                tour[0] = cities[startCityId];

                return new State(0, edges, tour, startCityId);
            }

            /**
             * Creates a child state from this state. Pass in the city that the child state will be going to.
             * Runs in O(n^2)
             */
            public State CreateChildState(City city, int from, int to)
            {
                // Create a new edges matrix
                int size = edges.Length;
                double[][] childEdges = new double[size][];
                for (int i = 0; i < size; i++)// O(n^2)
                {
                    childEdges[i] = new double[size];
                    Array.Copy(edges[i], childEdges[i], size);
                }

                // Create a new tour and add the city to it
                // TODO use increased memory by having mostly empty arrays, but slightly faster, or decrease memory by growing arrays as needed?
                City[] childTour = new City[tour.Length + 1];
                Array.Copy(tour, childTour, tour.Length);// O(n)
                childTour[tour.Length] = city;

                // Calculate the new bound. This is the current bound plus the cost of traveling to the city.
                double childBound = bound + edges[from][to];

                // Mark this city as visited. This is done by setting the from row and to column to infinity. This
                // effectively removes the edge between the two cities.
                // Mark that we can no longer go to the 'to' city
                for (int row = 0; row < size; row++)// O(n)
                {
                    childEdges[row][to] = double.PositiveInfinity;
                }
                // Mark that we can no longer visit the 'from' city
                for (int col = 0; col < size; col++)// O(n)
                {
                    childEdges[from][col] = double.PositiveInfinity;
                }
                // Remove the edges between the from and to cities
                childEdges[from][to] = double.PositiveInfinity;
                childEdges[to][from] = double.PositiveInfinity;

                return new State(childBound, childEdges, childTour, to);
            }

            /** Get this state's bound. O(1) */
            public double Bound()
            {
                return bound;
            }

            /** Get this state's depth in the search tree. O(1) */
            public int Depth()
            {
                return tour.Length;
            }

            /** Get the graph matrix for this state. O(1) */
            public double[][] Edges()
            {
                return edges;
            }

            /** Get the index of the city for this state. O(1) */
            public int CityId()
            {
                return cityId;
            }

            /**
             * Get the list of cities that make up the tour for this state.
             * Runs in O(n) since a copy of the internal tour array is returned.
             */
            public ArrayList Tour()
            {
                return new ArrayList(tour);
            }

            /**
             * Reduce the state's edges.
             * Runs in O(n^2)
             */
            public void Reduce()
            {
                int row,
                    col,
                    size = edges.Length;
                double min;

                // Reduce each row
                for (row = 0; row < size; row++)// O(n^2)
                {
                    min = double.PositiveInfinity;
                    for (col = 0; col < size; col++)// O(n)
                    {
                        if (edges[row][col] == 0)
                        {
                            // Row already reduced
                            min = 0;
                            break;
                        }
                        else if (min > edges[row][col])
                        {
                            // Found a new possible minimum
                            min = edges[row][col];
                        }
                    }
                    if (min != 0 && min != double.PositiveInfinity)
                    {
                        bound += min;
                        for (col = 0; col < size; col++)// O(n)
                        {
                            if (edges[row][col] != double.PositiveInfinity)
                                edges[row][col] -= min;
                        }
                    }
                }

                // Reduce each column
                for (col = 0; col < size; col++)
                {
                    min = double.PositiveInfinity;
                    for (row = 0; row < size; row++)
                    {
                        if (edges[row][col] == 0)
                        {
                            // Column already reduced
                            min = 0;
                            break;
                        }
                        else if (min > edges[row][col])
                        {
                            min = edges[row][col];
                        }
                    }
                    if (min != 0 && min != double.PositiveInfinity)
                    {
                        bound += min;
                        for (row = 0; row < size; row++)
                        {
                            if (edges[row][col] != double.PositiveInfinity)
                                edges[row][col] -= min;
                        }
                    }
                }
            }
        }

        /**
         * Priority queue implementation using a binary heap.
         * The heap takes, roughly, O(n) space, where n is the number of items in the queue.
         */
        private class BinaryHeapQueue
        {
            const int NO_CHILDREN = -1;
            /* Indicates how many bits to left shift the depth value to increase weight by depth. */
            const int DEPTH_SHIFT_FACTOR = 10; // 10 is equivalent to multiplication by 1024

            private int size;
            private int maxSize;
            private State[] heap;

            public BinaryHeapQueue(int initialCapacity)
            {
                heap = new State[initialCapacity];
                size = 0;
                maxSize = 0;
            }

            /**
             * Inserts the given state into the queue.
             * Runs in O(log|V|).
             */
            public void Insert(State s)
            {
                // Increase capacity if needed
                if (size >= heap.Length)
                {
                    // Increasing the capacity is O(n) but since only called once in a while becomes amoratized to O(1)
                    State[] newHeap = new State[size * 2];
                    Array.Copy(heap, newHeap, size);
                    heap = newHeap;
                }

                // Insert at the bottom and bubble up
                heap[size] = s;
                BubbleUp(size);

                // Increase the size and set the max size if needed
                size++;
                if (size > maxSize)
                {
                    maxSize = size;
                }
            }

            /**
             * Deletes the item with the smallest priority from the queue and returns that item.
             * Runs in O(log|V|).
             */
            public State DeleteMin()
            {
                if (Empty())
                {
                    // This shouldn't happen
                    Debug.WriteLine("DeleteMin called on empty queue!");
                    return null;
                }
                else
                {
                    // Remove the first element, sift down the remaining elements //
                    State element = heap[0];

                    // Replace the top node with the lowest node
                    heap[0] = heap[size - 1];

                    SiftDown(0);
                    size--;

                    return element;
                }
            }

            /** Returns true if the queue is empty. O(1) */
            public bool Empty()
            {
                return size == 0;
            }

            /** Marks the queue as empty. O(1) */
            public void MarkAsEmpty()
            {
                size = 0;
            }

            /** Returns the largest size the queue ever reached. O(1) */
            public double MaxSize()
            {
                return maxSize;
            }

            /**
             * Bubbles the node at the given index up until it is in the correct position in the tree.
             * Runs in O(log|V|). The tree's max height is log|V| and this might have to travel from the bottom to the top of the tree.
             */
            private void BubbleUp(int index)
            {
                // Get the parent node index, while the parent node's weight is greater than ours swap positions.
                int parent = (index - 1) / 2;
                while (index > 0 && Weight(parent) > Weight(index))
                {
                    Swap(parent, index);

                    // Move to the index's parent
                    index = parent;
                    parent = (index - 1) / 2;
                }
            }

            /**
             * Sifts the node at the given index down if needed. Continues sifting down until the tree is in a correct state.
             * Runs in O(log|V|). The tree height will be at most log|V|, and sift might have to travel from the top to the bottom.
             */
            private void SiftDown(int index)
            {
                // Get the smallest child node, while the child nodes weight is less than ours swap positions
                int minChild = MinChild(index);
                while (minChild != NO_CHILDREN && Weight(minChild) < Weight(index))
                {
                    Swap(minChild, index);

                    // Move to the next smallest child
                    index = minChild;
                    minChild = MinChild(index);
                }
            }

            /**
             * Swap the position of the two nodes in the heap and the lookup array.
             * Runs in O(1)
             */
            private void Swap(int node1, int node2)
            {
                State temp = heap[node1];
                heap[node1] = heap[node2];
                heap[node2] = temp;
            }

            /**
             * Returns the index of the child with the smallest priority.
             * Runs in O(1)
             */
            private int MinChild(int index)
            {
                int firstChild = (index * 2) + 1;
                int secondChild = firstChild + 1;
                if (firstChild >= size)
                {
                    // No children
                    return NO_CHILDREN;
                }
                else if (secondChild >= size || Weight(firstChild) < Weight(secondChild))
                {
                    // First child is the smallest or no second child
                    return firstChild;
                }
                else
                {
                    // Second child is the smallest (or equal)
                    return secondChild;
                }
            }

            /** Get the weight for the state at the given index. O(1) */
            private double Weight(int index)
            {
                return heap[index].Bound() - (heap[index].Depth() << DEPTH_SHIFT_FACTOR);
            }
        }
    }
}
