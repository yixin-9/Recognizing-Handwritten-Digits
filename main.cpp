// Description : Basic Classification using Supervised MachineLearning: Recognizing Handwritten Digits. A simple K-NN classier is created in this assignment using MNIST data set, the training data and testing data.
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <sstream>
#include <cstring>

# define IMG_SIZE 784
# define TRAINING_SIZE 5000
# define TESTING_SIZE 1500

using namespace std;

class CDigit {
    public :
    // get data
    unsigned char* Data(){
        return data;
    }
    // get or set label
    char& Label(){
        return label;
    }
    
    //Euclidean Distance/Similarity function
    double EuclideanDistance (const CDigit& i2);
    
    // Overloaded [] operator
    unsigned char& operator[](int i){
        return data[i];
    }
    
    private :
    //1D pixel array for a single 28 x28 b-bit image
    unsigned char data[IMG_SIZE];
    // single char label 0-9
    char label ;
    
}; //CDigit

double CDigit::EuclideanDistance (const CDigit& i2){
    double temp=0;
    for (int i=0; i<IMG_SIZE; i++) {
        auto inside=static_cast<int>(pow(data[i]-i2.data[i], 2));
        temp+= sqrt(inside);
    }
    return temp;
}

class Classifier {
    public :
    // Default constructor
    Classifier () : k(3) {
    }
   
    // Overloading constructor
    Classifier (int K):k(K){
    }
    
    // Provided : disable non - essential copying for this class (C ++11 and above )
    Classifier & operator=(const Classifier &) = delete ;
    Classifier (const Classifier &) = delete ;
    
    // Destructor
    ~ Classifier (){
        delete[] training_data;
        delete[] testing_data;
    }
    
    // Provided
    void LoadTrainingData ( char* filename )
    { ReadDigits ( filename , training_data , TRAINING_SIZE );}
   
    // Provided
    void LoadTestData ( char* filename )
    { ReadDigits ( filename , testing_data , TESTING_SIZE ); }
    
    // return the training data
    CDigit* TrainingData (){
        return training_data;
    }
    // return the testing data
    CDigit* TestingData (){
        return testing_data;
    }
    
    // Method to classify test data
    // calls EuclideanDistance(..) for each CDigit in testing_data
    void Classify (){
        int digit_frequency[10];
        for (int i = 0; i < TESTING_SIZE ; i++)
        {
            vector <pair <double , char >> dist_label ;
            for (int j = 0; j < TRAINING_SIZE ; j++)
            {
                double distance;
                 // Compute distance between test data image and each training image
                distance=training_data[j].EuclideanDistance(testing_data[i]);
                
                // Store the distance and label of training data in dist_label
                dist_label.push_back(make_pair(distance, training_data->Label()));
                
            }
            
            // Sort vector dist_label by distance in ascending order
            sort(dist_label.begin(), dist_label.end());

            // Compute the frequency of each digit in the top K entries
            for (int y=0; y<10; y++) {
                digit_frequency[y]=0;
            }
            for (int x=0; x<k; x++) {
             int a= (int) dist_label[x].second;
                digit_frequency[a]+=1;
            }

            // Get the max index (0 to 9) that corresponds to the label from digit_frequency
            int frequency=0;
            int max=0;
            for (int r=0; r<10; r++) {
                if (frequency<digit_frequency[r]) {
                    frequency=digit_frequency[r];
                    max=r;
            }
            }
            // Store estimated label in member variable
            classification.push_back ( ( char ) max );
        }
    }// classify
    
    
    // return the classification values
    vector <char>& Classification (){
        return classification;
    }
    
    private :
    int k = 3;
    CDigit * training_data = nullptr ;
    CDigit * testing_data = nullptr ;
    vector <char > classification ;
    
    // Private function called by LoadTrainingdata and LoadTestData
    // Read digits from file and store it in array
    void ReadDigits ( char* filename , CDigit* data ,int num_images ){
        ifstream myfile;
        myfile.open (filename);
        data= new CDigit [num_images];
        
        if(!myfile){
            cout<<"File read error.";
        }
        string temp;
        for (int y=0; y<num_images; y++) {
            for (int x=0; x<IMG_SIZE+1; x++) {
                if (x==0) {
                    getline(myfile, temp, ',');
                    data->Label()=temp[0];
                }else if (x!=IMG_SIZE){
                    getline(myfile, temp, ',');
                    data[y][x-1]=(unsigned char)atoi(temp.c_str());
                }else{
                    getline(myfile, temp);
                    data[y][x-1]=(unsigned char)atoi(temp.c_str());
                }
            }
        }
        
    }
};//Classifier

char* train_filename = "mnist_train_5000.csv";
char* test_filename = "mnist_test_1500.csv";

int main (int argc , char* argv[]){
    Classifier c (10);
    // load the test and training data stored in csv format
    c.LoadTestData ( test_filename );
    c.LoadTrainingData ( train_filename );
    
    // run the k- nearest neighbor classification
    c.Classify();
    //
    // compute the percentage of test images correctly labeled
    //
    int matches = 0;
    for (int i = 0; i < TESTING_SIZE ; i++)
    {
        if (c.Classification()[i] == c.TestingData()[i].Label())
            matches++;
    }
    cout << " matches / testing_size = " << ( double ) matches / TESTING_SIZE <<
     endl ;
    char stop ;
    cin >> stop ;
    
    return 0;
}// main
