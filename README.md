# Multi-Purpose MPC

1. [Introduction](#Introduction)
2. [Implementation Details](#ImplementationDetails)
   1. [MPC](##MPC)
3. [How-To](##How-To)
4. [Limitations and Outlook](#LimitationsOutlook)

## Introduction

In this repository you find an implementation of a multi-purpose Model Predictive Controller. The controller was implemented as a contribution to the [Automatic Control Project Course (EL2425)](https://www.kth.se/student/kurser/kurs/EL2425) at KTH Royal Institute of Technology, Stockholm. 
The developed algorithm was tested on a 1:10 RC car provided by [KTH Smart Mobility Lab](https://www.kth.se/dcs/research/control-of-transport/smart-mobility-lab/smart-mobility-lab-1.441539). The test scenarios comprised the following three tasks:

1. Reference Path Tracking
2. Time-Optimal Driving
3. Obstacle Avoidance

The controller is implemented in a way that enables its application to all three tasks by merely tuning the weight matrices of the underlying optimization problem. The illustration below shows the obstacle avoidance task in simulation.

The rest of this readme is structured as follows. In [Section 2](##Components) we will present an overview of the entire system and discuss all fundamental components of the implementation in detail. In [Section 3](##How-To) we will provide guidelines for using the implementation in simulation and practice. [Section 4](##Limitations) will be dedicated to analyzing limitations of the current version of the controller and outline potential extensions of the implementation.

## Implementation Details


## How-To

## Limitations and Outlook
