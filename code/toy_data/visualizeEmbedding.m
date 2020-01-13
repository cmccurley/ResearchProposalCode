function [] = visualizeEmbedding(data, embData, parameters)
%{ 
***********************************************************************
    *  File:  visualizeEmbedding.m
    *  Name:  Connor McCurley
    *  Date:  2018-11-12
    *  Desc:  
    *  Inputs:
    *  Outputs:
**********************************************************************
%}

figure();
scatter(embData(1,:),embData(2,:),20,'b','filled');

showExamples()


end