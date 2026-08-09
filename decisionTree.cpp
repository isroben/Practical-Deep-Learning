#include <iostream>
#include <vector>
using namespace std;

class Sample{
public:
    int age;
    int salary;
    int label;

    Sample(int age, int salary, int l){
        this->age = age;
        this->salary = salary;
        this->label = l;
    }
};


class Node{
public:
    int featureIndex; // which feature to split on
    double threshold; // Split value

    int prediction; // used only for leaf nodes
    bool isLeaf;

    Node* left;
    Node* right;

    Node(){
        featureIndex = -1;
        threshold = 0;

        isLeaf = false;
        prediction = -1;

        left = NULL;
        right = NULL;
    }
};

class Split{
public:
    int featureIndex;
    int threshold;
    double score;

    Split(){
        featureIndex = -1;
        threshold = 0;
        score = 999999; // start with bad score
    }
};

class DecisionTree{
private:
    Node* root;
    
    // Recursive prediction
    int predict(Node* node, int age, int salary){
        if(node->isLeaf){
            return node->prediction;
        }
        if(node->featureIndex == 0){
            if(age <= node->threshold) return predict(node->left, age, salary);
            else return predict(node->right, age, salary);
        }
        return -1;
    }

    double gini(vector<Sample>& data){
        int countNo = 0;
        int countYes = 0;

        for(const Sample& sample: data){
            if(sample.label == 0){
                countNo++;
            }
            else{
                countYes++;
            }
        }

        double total = data.size();

        double pNo = countNo / total;
        double pYes = countYes / total;

        return 1.0 - (pNo*pNo + pYes*pYes);
    }

    double scoreSplit(vector<Sample>& left, vector<Sample>& right){
        int total = left.size() + right.size();

        double leftWeight = (double) left.size() / total;
        double rightWeight = (double) right.size() / total;

        return leftWeight*gini(left) + rightWeight*gini(right);
    }

    int majorityClass(vector<Sample>& data){
        int countNo = 0;
        int countYes = 0;

        for(const Sample& sample: data){
            if(sample.label == 0){
                countNo++;
            }
            else countYes++;
        }
        if (countYes > countNo) return 1;
        
        return 0;
    }


    Split findBestSplit(vector<Sample>& data){
        Split best;

        // Try every feature
        for(int feat = 0; feat < 2; feat++){
            for(const Sample& sample: data){
                int threshold;

                if(feat == 0) threshold = sample.age;
                else threshold = sample.salary;

                vector<Sample> left;
                vector<Sample> right;

                // Divide the dataset
                for(const Sample& s: data){
                    int value;

                    if(feat == 0) value = s.age;
                    else value = s.salary;
                    
                    if(value <= threshold) left.push_back(s);
                    else right.push_back(s);
                }
                if(left.empty() || right.empty()) continue;

                double score = scoreSplit(left, right);

                if(score < best.score){
                    best.featureIndex = feat;
                    best.threshold = threshold;
                    best.score = score;
                }
            }
        }
        return best;
    }

    Node* buildTree(vector<Sample>& data){
        Node* node = new Node();

        bool same = true;

        for(int i=1; i<data.size(); i++){
            if(data[i].label != data[0].label){
                same = false;
                break;
            }
        }
        // base case
        if(same){
            node->isLeaf = true;
            node->prediction = data[0].label;
            return node;
        }

        Split best = findBestSplit(data);

        if(best.featureIndex == -1){
            node->isLeaf = true;
            node->prediction = majorityClass(data);

            return node;
        }

        node->threshold = best.threshold;
        node->featureIndex = best.featureIndex;

        vector<Sample> leftData;
        vector<Sample> rightData;

        for(const Sample& s: data){
            int val;
            if(best.featureIndex == 0) val = s.age;
            else val = s.salary;

            if(val <= best.threshold){
                leftData.push_back(s);
            }
            else{
                rightData.push_back(s);
            }
        }

        node->left = buildTree(leftData);
        node->right = buildTree(rightData);

        return node;

    }

    int predict(Node* node, int age, int salary){
        if(node->isLeaf){
            return node->prediction;
        }

        int value;

        if(node->featureIndex == 0){
            value = age;
        }
        else value = salary;

        if(value <= node->threshold){
            return predict(node->left, age, salary);
        }
        else{
            return predict(node->right, age, salary);
        }
    }


public:
    DecisionTree(){
        root = NULL;
    }
    void train(vector<Sample>& data){
        root = buildTree(data);
    }
    int predict(int age, int salary){
        return predict(root, age, salary);
    }

};




int main()
{
    vector<Sample> data;


    data.push_back(Sample(22,25000,0));
    data.push_back(Sample(25,30000,0));
    data.push_back(Sample(28,40000,0));

    data.push_back(Sample(35,60000,1));
    data.push_back(Sample(40,70000,1));
    data.push_back(Sample(45,80000,1));



    DecisionTree tree;


    tree.train(data);



    int result = tree.predict(27,35000);



    if(result == 1)
        cout<<"Buy Laptop";
    else
        cout<<"Don't Buy Laptop";


    return 0;
}