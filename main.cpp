#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;

int computeUpperBound(const vector<vector<int>>& edgesWeights, const vector<int>& s, const vector<int>& c, const vector<int>& storages,
                      const vector<int>& computations, const vector<int>& fixedNode);

void SOA(const vector<vector<int>>& edgesWeights, const vector<int>& s, const vector<int>& c, const vector<int>& storages,
         const vector<int>& computations, const vector<int>& fixedNode, int n, int k, int upperBound, vector<vector<int>>& x);

double LLBP(const vector<vector<int>>& edgesWeight, const vector<int>& s, const vector<int>& c, const vector<int>& storages,
            const vector<int>& computations, const vector<int>& fixedNodes, vector<vector<vector<double>>>& lanna1, vector<vector<vector<double>>>& miu1,
vector<vector<vector<int>>>& z, vector<vector<int>>& x);

bool comp(const pair<double, int>& p1, const pair<double, int>& p2) {
    //return p1.first > p2.first;
    return p1.first < p2.first;
}

int main() {
    /*********variables************/
    //input variable
    int n = 0, k = 0;
    //open ifstream
    ifstream in("input.txt");

    in>>n>>k;
    vector<vector<int>> edgesWeights(n, vector<int>(n, 0));
    vector<int> s(n, 0);
    vector<int> c(n, 0);
    vector<int> storages(k, 0);
    vector<int> computations(k, 0);
    vector<int> fixedNodes(k, 0);
    //compute variable
    vector<vector<int>> x(n, vector<int>(k, 0));
    vector<int> where(n, 0);
    int upperBound = 0, cost = 0;

    /************read input data******************/
    //read edgeWeights
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            in>>edgesWeights[i][j];
        }
    }
    //read node storage size
    for(int i = 0; i < n; i++) {
        in>>s[i];
    }
    //read node computation size
    for(int i = 0; i < n; i++) {
        in>>c[i];
    }
    //read datacenter storage resources
    for(int i = 0; i < k; i++) {
        in>>storages[i];
    }
    //read datacenter computation resources
    for(int i = 0; i < k; i++) {
        in>>computations[i];
    }
    //read fixedNodes
    for(int i = 0; i < k; i++) {
        in>>fixedNodes[i];
    }
    //close ifstream
    in.close();

    //test read
    /*
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cout<<edgesWeights[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    for(int i = 0; i < n; i++) {
        cout<<s[i]<<" ";
    }
    cout<<endl;
    for(int i = 0; i < n; i++) {
        cout<<c[i]<<" ";
    }
    cout<<endl;
    cout<<endl;
    for(int i = 0; i < fixedNodes.size(); i++) {
        cout<<fixedNodes[i]<<" ";
    }
    cout<<endl;
     */
    /*****************computation*****************/
    //get upper bound
    upperBound = computeUpperBound(edgesWeights, s, c, storages, computations, fixedNodes);

    //subgradient optimization algorithm
    SOA(edgesWeights, s, c, storages, computations, fixedNodes, n, k, upperBound, x);

    /********************output*******************/
    //compute variable where
    for(int i = 0; i < n; i++) {
        for(int t = 0; t < k; t++) {
            if(x[i][t]) {
                where[i] = t;
                break;
            }
        }
    }
    //compute cutCost
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(edgesWeights[i][j] && where[i] != where[j]) {
                cost += edgesWeights[i][j];
            }
        }
    }
    cost = cost/2;
    cout<<"cost : "<<cost<<endl;
    /*
    ofstream out("output.txt");
    int index = -1;
    for(int i = 0; i < n; i++) {
        for(int t = 0; t < k; t++) {
            if(x[i][t] == 1) {
                index = t;
                break;
            }
        }
        cout<<index<<" ";
    }
    cout<<endl;
    out.close();
    */
    return 0;
}

int computeUpperBound(const vector<vector<int>>& edgesWeights, const vector<int>& s, const vector<int>& c, const vector<int>& storages,
                      const vector<int>& computations, const vector<int>& fixedNode) {
    int cutCost = 0, n = s.size(), k = storages.size();
    vector<vector<int>> connects;
    vector<vector<int>> connectsWeight;
    vector<int> totalStorage(storages);
    vector<int> totalComputate(computations);
    vector<int> where(n, -1);
    vector<vector<int>> dependency(k, vector<int>(n, 0));

    connects.resize(n);
    connectsWeight.resize(n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(edgesWeights[i][j] > 0) {
                connects[i].push_back(j);
                connectsWeight[i].push_back(edgesWeights[i][j]);
            }
        }
    }
/*
    vector<int> totalStorage(storages);
    vector<int> totalComputate(computations);
    vector<int> where(n, -1);
  */
    for(int i = 0; i < k; i++) {//we think the number of fixedNodes is the same as the number of datacenters(because every datacenters has some fixed nodes, and we contract tnem to one)
        where[i] = i;
        totalStorage[i] -= s[i];
        totalComputate[i] -= c[i];
    }

    //compute the dependency of every nodes to every datacenters
    //vector<vector<int>> dependency(k, vector<int>(n, 0));
    for(int i = k; i < n; i++) {
        //if(where[i] != -1) continue;
        for(int j = 0; j < connects[i].size(); j++) {//be careful, it's j++, not i++
            int place = where[connects[i][j]];
            if(place != -1) {
                dependency[place][i] += connectsWeight[i][j];
            }
        }
    }

    int chooseNode = -1, chooseDatacenter = -1, maxValue = INT32_MIN, cnt = n-k;
    while(cnt > 0) {
        //choose the heaviest dependency node and the responce datacenter
        chooseNode = -1;
        chooseDatacenter = -1;
        maxValue = INT32_MIN;
        for(int t = 0; t < k; t++) {
            for(int i = k; i < n; i++) {
                if(where[i] != -1) continue;
                if((dependency[t][i] > maxValue && totalStorage[t] > s[i] && totalComputate[t] > c[i])
                   || (dependency[t][i] == maxValue && totalStorage[t] > s[i] && totalComputate[t] > c[i]
                       && totalStorage[t] > totalStorage[chooseDatacenter] && totalComputate[t] > totalComputate[chooseDatacenter])) {//here, we should do more charges
                    chooseNode = i;
                    chooseDatacenter = t;
                    maxValue = dependency[t][i];
                }
            }
        }
        if(chooseNode == -1) break;
        //modify the choose node
        where[chooseNode] = chooseDatacenter;
        totalStorage[chooseDatacenter] -= s[chooseNode];
        totalComputate[chooseDatacenter] -= c[chooseNode];
        cnt--;
        for(int t = 0; t < k; t++) {
            dependency[t][chooseNode] = 0;
        }
        //modify the connected node
        for(int i = 0; i < connects[chooseNode].size(); i++) {
            if(where[i] != -1) continue;
            dependency[chooseDatacenter][i] += connectsWeight[chooseNode][i];
        }
    }

    //for the rest nodes, we random select the datacenter
    if(cnt > 0) {
        for(int i = k; i < n; i++) {
            if(where[i] == -1) {
                for(int t = 0; t < k; t++) {
                    if(totalStorage[t] > s[i] && totalComputate[t] > c[i]) {
                        where[i] = t;
                        totalStorage[t] -= s[i];
                        totalComputate[t] -= c[i];
                        break;
                    }
                }
            }
        }
    }

    //compute cutCost
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if(edgesWeights[i][j] != 0 && where[i] != where[j]) {
                cutCost += edgesWeights[i][j];
            }
        }
    }
    cutCost /= 2;
    cout<<"upperBound : "<<cutCost<<endl;
    for(int i = 0; i < n; i++) {
        cout<<where[i]<<" ";
    }
    cout<<endl;

    return cutCost;
}

void SOA(const vector<vector<int>>& edgesWeights, const vector<int>& s, const vector<int>& c, const vector<int>& storages,
         const vector<int>& computations, const vector<int>& fixedNodes, int n, int k, int upperBound, vector<vector<int>>& x) {
    //int z[n][n][k] = {0};//edge partition
    vector<vector<vector<int>>> z(n, vector<vector<int>>(n, vector<int>(k, 0)));//it is ok
    //int lanna1[n][n][k] = {0};
    vector<vector<vector<double>>> lanna1(n, vector<vector<double>>(n, vector<double>(k, 0)));
    //int miu1[n][n][k] = {0};
    vector<vector<vector<double>>> miu1(n, vector<vector<double>>(n, vector<double>(k, 0)));

    vector<vector<int>> recordX(n, vector<int>(k, 0));
    vector<vector<vector<int>>> recordZ(n, vector<vector<int>>(n, vector<int>(k, 0)));

    double maxValue = INT32_MIN, llbp = 0.0;
    double pi = 2;
    int iterations = 500;

    int yita1[n][n][k] = {0};//type int
    int yita2[n][n][k] = {0};
    double delta = 0;

    int cnt = 0, chooseIteration = 0;//test
    do{
        //reset x, z
        for(int i = 0; i < n; i++) {
            for(int t = 0; t < k; t++) {
                x[i][t] = 0;
            }
        }
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int t = 0; t < k; t++) {
                    z[i][j][t] = 0;
                }
            }
        }

        //solve x and z and compute llbp
        llbp = LLBP(edgesWeights, s, c, storages, computations, fixedNodes, lanna1, miu1, z, x);
        //test llbp
        cout<<"llbp : "<<llbp<<endl;

        //test x and z
        /*
        for(int i = 0; i < n; i++) {
            for(int t = 0; t < k; t++) {
                cout<<x[i][t]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int t = 0; t < k; t++) {
                    cout<<z[i][j][t]<<" ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        */
        //test lanna1
        /*
        cout<<"lanna1 : "<<endl;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int t = 0; t < k; t++) {
                    cout<<lanna1[i][j][t]<<" ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout<<endl;
        */

        //compute subgradient, step size and update multipliers for lanna1, miu1
        //compute yita1 and yita2
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int t = 0; t < k; t++) {
                    yita1[i][j][t] = x[i][t] - x[j][t] - z[i][j][t];
                    yita2[i][j][t] = x[j][t] - x[i][t] - z[i][j][t];
                }
            }
        }

        //compute step size
        int sum = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int t = 0; t < k; t++) {
                    if(yita1[i][j][t]) sum += yita1[i][j][t] * yita1[i][j][t];
                    if(yita2[i][j][t]) sum += yita2[i][j][t] * yita2[i][j][t];
                }
            }
        }
        delta = pi*(upperBound - llbp)/sum;

        //test delta
        cout<<"delta : "<<delta<<endl;

        //update lagrangian multipliers
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int t = 0; t < k; t++) {
                    lanna1[i][j][t] = max(0.0, lanna1[i][j][t] + delta * yita1[i][j][t]);
                    miu1[i][j][t] = max(0.0, miu1[i][j][t] + delta * yita2[i][j][t]);
                }
            }
        }

        //remember best lower bound
        //if(maxValue < llbp && llbp < upperBound) {
        if(maxValue < llbp) {
            chooseIteration = iterations;
            maxValue = llbp;
            //preserve the value of x and z
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    for(int t = 0; t < k; t++) {
                        recordZ[i][j][k] = z[i][j][k];
                    }
                }
            }
            for(int i = 0; i < n; i++) {
                for(int t = 0; t < k; t++) {
                    recordX[i][t] = x[i][t];
                }
            }
            cnt = 0;
        }
        else {
            cnt++;
        }

        //reduce agility
        if(cnt > 30) {
            pi = pi/2;
            if(pi <= 0.005) break;
            cnt = 0;
        }
    } while(abs(upperBound - llbp) >= 0.5 && --iterations);
    //} while(upperBound > llbp && --iterations);

    for(int i = 0; i < n; i++) {
        for(int t = 0; t < k; t++) {
            x[i][t] = recordX[i][t];
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int t = 0; t < k; t++) {
                z[i][j][t] = recordZ[i][j][t];
            }
        }
    }
    //test x and z constraints
    /*
    for(int i = 0; i < n; i++) {
        for(int t = 0; t < k; t++) {
            cout<<x[i][t]<<" ";
        }
        cout<<endl;
    }
    */
    /*
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int t = 0; t < k; t++) {
                if(z[i][j][t] >= x[i][t] - x[j][t] && z[i][j][t] >= x[j][t] - x[i][t]) {
                    cout<<1<<" ";
                }
                else {
                    cout<<0<<" ";
                }
            }
            cout<<endl;
        }
        cout<<endl;
    }
    */
    //test the best x
    /*
    cout<<endl;
    vector<int> realStorages(k, 0);
    vector<int> realComputation(k, 0);
    for(int i = 0; i < n; i++) {
        for(int t = 0; t < k; t++) {
            if(x[i][t] == 1) {
                realStorages[t] += s[i];
                realComputation[t] += c[i];
            }
            cout<<x[i][t]<<" ";
        }
        cout<<endl;
    }

    cout<<"maxValue : "<<maxValue<<endl;

    //test datacenter constraints
    for(int t = 0; t < k; t++) {
        cout<<realStorages[t]<<" "<<realComputation[t]<<endl;
    }
    */
    /*
    cout<<"iteration : "<<iterations<<endl;
    cout<<"chooseIteration : "<<chooseIteration<<endl;
    */
}

double LLBP(const vector<vector<int>>& edgesWeight, const vector<int>& s, const vector<int>& c, const vector<int>& storages,
            const vector<int>& computations, const vector<int>& fixedNodes, vector<vector<vector<double>>>& lanna1, vector<vector<vector<double>>>& miu1,
            vector<vector<vector<int>>>& z, vector<vector<int>>& x) {
    double cost1 = 0.0, cost2 = 0.0, tmpValue = 0.0, minValue = INT32_MAX, sum = 0.0;
    int n = x.size(), k = x[0].size(), minIndex = -1;
    vector<int> storageCapacities(storages);
    vector<int> computationCapacities(computations);
    vector<pair<double, int>> vp;

    //compute z and cost1
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int t = 0; t < k; t++) {
                tmpValue = edgesWeight[i][j]/2.0 - lanna1[i][j][t] - miu1[i][j][t];//shit
                if(tmpValue > 0.0) {
                    z[i][j][t] = 0;
                }
                else {
                    z[i][j][t] = 1;
                    cost1 += tmpValue;
                }
            }
         }
    }

    //compute x and cost2
    /****************************rewrite****************************/
    for(int i = 0; i < n; i++) {
        if(i < k) {//the front k nodes are fixed nodes
            sum = 0.0;
            for(int j = 0; j < n; j++) {
                sum += (lanna1[i][j][i] - miu1[i][j][i] + miu1[j][i][i] - lanna1[j][i][i]);
            }
            minIndex = i;
            minValue = sum;
        }
        else {
            vp.clear();
            for(int t = 0; t < k; t++) {
                sum = 0.0;
                for(int j = 0; j < n; j++) {
                    sum += (lanna1[i][j][t] - miu1[i][j][t] + miu1[j][i][t] - lanna1[j][i][t]);
                }
                vp.push_back(make_pair(sum, t));
            }
            sort(vp.begin(), vp.end(), comp);
            //test sort
            /*
            cout<<endl;
            for(int i = 0; i < vp.size(); i++) {
                cout<<vp[i].first<<" ";
            }
            cout<<endl;
            */
            for(int j = 0; j < vp.size(); j++) {
                if(storageCapacities[vp[j].second] >= s[i] && computationCapacities[vp[j].second] >= c[i]) {
                    minIndex = vp[j].second;
                    minValue = vp[j].first;
                    break;//can not loss
                }
            }
        }

        x[i][minIndex] = 1;
        cost2 += minValue;
        storageCapacities[minIndex] -= s[i];
        computationCapacities[minIndex] -= c[i];
    }

    //test cost1 and cost2
    cout<<cost1<<" "<<cost2<<endl;

    return (cost1 + cost2);
}
