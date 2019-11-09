## Graduation Final Project 

Project being developed at Universidade Federal do Rio de Janeiro by Fabiana Ferreira Fonseca as the final project of the Electronic and Computer Engineering undergraduate degree.


### Setting up the environment

It requires to have OpenCV 4+ with the extras modules (opencv_contrib) and CMake installed. If you dont't have CMake installed, you can use the following command:
`sudo apt install cmake`

* Cloning the repositories
  
    Clone [opencv](https://github.com/opencv/opencv) and [opencv extra modules](https://github.com/opencv/opencv_contrib) repositories
    
    `$ git clone https://github.com/opencv/opencv.git`
    
    `$ git clone https://github.com/opencv/opencv_contrib`
    
* Make sure that you have the correct dependencies
    
    `$ sudo apt-get update`
    
    `$ sudo apt install ubuntu-restricted-extras`
    
    `$ sudo apt-get install gtk+2.0`
    
* Building OpenCV

    Create a directory to build OpenCV
    
    ```
    $ mkdir <opencv_build_directory>
    $ cd <opencv_build_directory>
    ```

    Use flags to enable non-free modules when using cmake

    ```
    $ cmake -DOPENCV_GENERATE_PKGCONFIG=YES -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>
    $ make -j5
    $ sudo make install
    ```
