function fileSorted=dicomreaddir(dirName)

% if you want a gui for asking which directory
% uncomment line below
dirName=uigetdir();

if dirName==0
    fileSorted=[];
    disp(' Empty directory ');
else
    D=dir(dirName);
    [nooffilesf garb]=size(D);

    fileX = [];
    for i=3:nooffilesf  % this assumes that the first and second are . and ..
        % uncomment the if statement below if you want to select only
        % filenames starting with some characters say 'im' in this example
        % similarly we can have a condition to check if the files end in
        % say .dcm 
        if (strfind(D(i).name, 'dcm')==1)
            fileX(i-2).name = D(i).name;
        end    
    end

    [garb nooffiles]=size(fileX);
    
    InstanceN =[]; 
    for i=1:nooffiles
        fullfilename = strcat(dirName,'/',fileX(i).name);
        fileinfo=dicominfo(fullfilename);
        InstanceN= [InstanceN; fileinfo.InstanceNumber];
    end
    
    [XTemp, Ind]=sort(InstanceN);

    fileSorted=[];
    for i=1:nooffiles
        fileSorted(i).name = strcat(dirName,'/',fileX(Ind(i)).name);
    end
    
    %fileSorted(:).name
   
end