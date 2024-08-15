%% 
f=imread('moon.jfif');
figure(1);
imshow(f)

red = f(:,:,1);
imshow(red)

F = fft2(red);
figure(2)
imshow(log(abs(F)),[]);
figure(3)
imshow(log(abs(fftshift(F))),[]);

%%
sz = 50;
f=zeros(sz);
angle=0;
frequency_X = 0.5/2;
frequency_Y=frequency_X;
for i=1:sz 
    for j=1:sz
        f(i,j)=cos(2*pi*(j*frequency_X*cos(angle*pi/180)+i*frequency_Y*sin(angle*pi/180)));
    end
end
figure(1)
subplot(2,1,1)
plot(f(:,1))
subplot(2,1,2)
plot(f(1,:))
figure(2)
imshow(f,[])
figure(3)
mesh(f)
F=fft2(f);
figure(4)
imshow(log(abs(fftshift(F))),[]);
figure(5)
mesh(abs(fftshift(F)))
%%
sz = 256;

f=zeros(256);


for i=1:sz 
    for j=1:sz
        x = j-sz/2;
        y = i-sz/2;
        if (abs(x)<16) & (abs(y)<16)
         f(i,j)=1;
        end
    end
end
figure(1)
subplot(1,1,1)
imshow(f)

hsize=21;
h=ones(hsize)/(7*7);
vari=0.10;
for i=1:hsize 
    for j=1:hsize
        x = j-hsize/2;
        y = i-hsize/2;
         h(i,j)=exp(-(x^2+y^2)/vari);
    end
end
h=h/sum(h,"all");
figure(2)
mesh(h)

figure(3)
mesh(abs(fftshift(fft2(h,sz,sz))))

%mesh(h)
%mesh(f)

g=conv2(f,h);
figure(4);
imshow(g,[]);

%%

h=[-1,0,1;-2,0,2;-1,0,1]

figure(4)
mesh(abs(fftshift(fft2(h,sz,sz))))

g=conv2(f,h);
figure(5);
imshow(g,[]);
figure(6)
g=xcorr2(f,h);
imshow(g,[]);

%%
hy=[-1,-2,-1;0,0,0;1,2,1]

figure(4)
mesh(abs(fftshift(fft2(hy,sz,sz))))

g=conv2(f,hy);
figure(5);
imshow(g,[]);
figure(6)
mesh(g)




