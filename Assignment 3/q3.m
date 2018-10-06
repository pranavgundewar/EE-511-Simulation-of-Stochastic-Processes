% x = table2array(nipsdata);
x = csvread(nipsdata);
eva=evalclusters(x,'kmeans','CalinskiHarabasz','KList',[1:6]);
disp(eva);
idx = kmeans(x, 2); 
y = idx; first part of the problem ends here
% second part to group the document ids
j=1;
k=1;
v = sheet123;
for i = 1:700

        if y(i) == 1

            e(j) = v(i);
            j = j+1;

        else

                f(k) = v(i);
                k = k+1;

        end
end
            fprintf('The document ids of cluster 1 are:\n');
            disp(e);
            fprintf('The document ids of cluster 2 are:\n');
            disp(f);
