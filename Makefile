main:
	g++ -O3 -D LOG=true -I eigen-3.4.0/ extremeLearning.cpp main.cpp -o main

clean:
	rm main

debug:
	g++ -D LOG=true -D TUX_COMPARISON=true -I eigen-3.4.0/ extremeLearning.cpp main.cpp -o main
