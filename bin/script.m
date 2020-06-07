%% get dicom array

% set argument as directory that contains dicom files.
dirName=uigetdir();

D=dir(dirName);
[nooffilesf, ~]=size(D);

inds = zeros(nooffilesf, 1);

for i=1:nooffilesf
    filename = D(i).name;
    if length(D(i).name)>4
        inds(i) = prod((D(i).name(end-3:end))=='.dcm');
    end
end

lstfiles = cell(length(inds(inds==1)), 1);
% lstfiles = cell(lstfiles);

p=1;
for i=1:nooffilesf
    if inds(i) == 1
        lstfiles{p} = char(D(i).name);
        p = p+1;
    end
end


% get dimensions of dicom file
path = strcat(dirName,'/',char(lstfiles(1)));
sz = size(dicomread(path));

array = zeros(sz(1), sz(2), length(lstfiles));

for i=1:length(lstfiles)
    path = strcat(dirName,'/',char(lstfiles(i)));
    array(:,:,i) = dicomread(path);
end

xlim = sz(1); ylim=sz(2);

%% try plotting contours

% image(array(:,:,100), 'CDataMapping', 'scaled')
colormap jet
cmap = colormap;
cm = brighten(jet(length(cmap)),-.5);
colormap(cm)
contourslice(array,[],[],[1,50,100,110],8);
view(3)
axis tight
% axis ij
daspect([1,1,1])

%% try isocaps
% too heavy
D = array;
D(D<100)=nan;
D(:,1:60,:) = [];
disp('starting p1')
p1 = patch(isosurface(D, 5),'FaceColor','blue',...
    'EdgeColor','none');
disp('p1 complete')
p2 = patch(isocaps(D, 5),'FaceColor','interp',...
    'EdgeColor','none');
disp('p2 complete')
view(3); axis tight; daspect([1,1,.4])
colormap(gray(100))
camlight left; camlight; lighting gouraud
isonormals(D,p1)

%% try isocaps 2

D = array;
D(D<100)=nan;

figure
colormap(cmap)
Ds = smooth3(D, 'gaussian', 5);
disp('start isosurface patch')
hiso = patch(isosurface(Ds,1000),...
   'FaceColor','blue',...
   'FaceAlpha', 0.7, ...
   'EdgeColor','none');
   isonormals(Ds,hiso)

disp('start isocap patch')
hcap = patch(isocaps(D,100),...
    'FaceColor','interp',...
    'EdgeColor','none');

view(35,30) 
axis tight 
daspect([1,1,.4])

lightangle(45,30);
lighting gouraud
hcap.AmbientStrength = 0.6;
hiso.SpecularColorReflectance = 0;
hiso.SpecularExponent = 50;

%% try many slices

D = array;
D(D<100)=nan;
h = slice(D,[],[],[1:20:100]);
set(h,'EdgeColor','none',...
'FaceColor','interp',...
'FaceAlpha','interp')
axis tight 
daspect([1,1,.4])
alpha('color')
alphamap('rampdown')
alphamap('decrease',0.3)
colormap bone

%% try drawing midplane

