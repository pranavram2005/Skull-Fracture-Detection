#Import Required Packages
import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render,redirect
from ultralytics import YOLO
import base64
import glob
from random import randint
from io import BytesIO
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
from skimage.transform import resize
from IPython.display import Image as show_gif
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.transform import resize
import warnings
warnings.simplefilter("ignore")

#FileSystemStorage is used to store the given files in the given location(Here it is media)
class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name
#Class used in Image to GIF conversion which will be called in, line 120.
class ImgtoGIF:
  def __init__(self,
               size=(500, 500),
                 xy_text=(80, 30),
                 dpi=100,
                 cmap='CMRmap'):
    self.fig = plt.figure()
    self.fig.set_size_inches(size[0] / dpi, size[1] / dpi)
    self.xy_text = xy_text
    self.cmap = cmap
    #Plotting starts from here.
    self.ax = self.fig.add_axes([0, 0, 1, 1])
    self.ax.set_xticks([])
    self.ax.set_yticks([])
    self.images = []
  #Adds the images to one single list for animation.
  def add(self,image,label,with_mask=False):
    plt.set_cmap(self.cmap)
    plt_img = self.ax.imshow(image, animated=True)
    plt_text = self.ax.text(*self.xy_text,label,color="red")
    to_plot=[plt_img,plt_text]
    self.images.append(to_plot)
    plt.close()
  #Saves the animation to the given file name.
  def save(self, filename, fps):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=fps)

#3d reconstruction
#normalize the given array for 3D reconstruction.
def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))
#Used to transform the colours more uniformly.
def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr-mean)*fac + mean

#Detecting fractures in the skull.

def detection(request):
    fss = CustomFileSystemStorage()
    if request.method == 'POST':
        uploaded_image = request.FILES['file']#get the image from the html page through request and post method.
        print("Name", uploaded_image.file)
        _image = fss.save(uploaded_image.name, uploaded_image)#now save the image in media
        path = str(settings.MEDIA_ROOT) + "/" + uploaded_image.name
        # image details
        image_url = fss.url(_image)
        image = cv2.imread(path)#call and read the image
        model = YOLO("D:\\skull_gui\\skull\\myapp\\best.pt")#call the pretrained model though yolo
        detect_result = model(image) 
        detect_img = detect_result[0].plot()
        detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)#predict the fractures for selected image.
        _, img_encoded = cv2.imencode('.png', detect_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        context = {'detected_image': img_base64}#encode the preicted image and pass it through the render as "context".
        return render(request, 'detection.html', context)
    return render(request, 'show.html')
    #3D reconstruction of the image.
    #GIF
def reconstruction(request):
    fs = CustomFileSystemStorage(location="uploads")#save in uploads 
    if request.method == 'POST':
        count=0
        uploaded_files = request.FILES.getlist('files')#get the files through post method.
        for uploaded_file in uploaded_files:
            filename = fs.save(uploaded_file.name, uploaded_file)
            count+=1   
        instance=["0" for j in range(count)]#get the list of image paths in correct order.
        path_x = "uploads/*.dcm"
        start = len("uploads/CT000")
        end = len(".dcm")
        path_to_slices = glob.glob(path_x)
        for i in path_to_slices:
            ds = pydicom.read_file(i).InstanceNumber
            a=int(ds)-1
            instance[a]=i  
        print(instance) #instance is the list having the correct order.

        sample_data_gif = ImgtoGIF()#calling the ImgtoGIF class.
        label = "upload"
        filename = f'D:\\skull_gui\\skull\\assets\\upload_3d_2d.gif'#please try to change the path according to your file path
        for i in range(len(instance)):
            image = pydicom.read_file(instance[i]).pixel_array
            sample_data_gif.add(image, label=f'{label}_{str(i)}')
        sample_data_gif.save(filename, fps=3)#add and save the gif file using the list of images
        #3D
        tensor = np.zeros((512, 512, len(instance)))
        for i in range(len(instance)):
            image = pydicom.read_file(instance[i]).pixel_array
            tensor[:,:,i] = image
        arr = tensor    #tensor is the matrix containing the 3d images. It is generated using the above code.
        transformed = np.clip(scale_by(np.clip(normalize(arr)-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
        IMG_DIM = 50
        resized = resize(transformed, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')#use the scale_by and normalize functions from above.
        def explode(data):#we are now going to enlarge the image using explode and expand_coordinates.
            shape_arr = np.array(data.shape)
            size = shape_arr[:3]*2 - 1
            exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
            exploded[::2, ::2, ::2] = data
            return exploded

        def expand_coordinates(indices):#the enlarging helps in showing magnified and clear image.
            x, y, z = indices
            x[1::2, :, :] += 1
            y[:, 1::2, :] += 1
            z[:, :, 1::2] += 1
            return x, y, z

        def plot_cube(cube, angle=320):#now we eill plot using above functions.
            cube = normalize(cube)
    
            facecolors = cm.viridis(cube)
            facecolors[:,:,:,-1] = cube
            facecolors = explode(facecolors)
    
            filled = facecolors[:,:,:,-1] != 0
            x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

            fig = plt.figure(figsize=(30/2.54, 30/2.54))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(30, angle)
            ax.set_xlim(right=IMG_DIM*2)
            ax.set_ylim(top=IMG_DIM*2)
            ax.set_zlim(top=IMG_DIM*2)
    
            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
    
            # Encode the image as base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
    
            return image_base64
        cube = np.copy(resized)

        for x in range(0, IMG_DIM):
            for y in range(0, IMG_DIM):
                for z in range(max(x+5, 0), IMG_DIM):#z=x+5 is the cut of the image.
                    cube[x, y, z] = 0
        plot_image = plot_cube(cube, angle=200)#we are calling the plot_cube function for plotting.



        return render(request,"result.html",{'plot_image': plot_image})  
    return render(request,'reconstruction.html')

def skull_home(request):
    return render(request,"skull_home.html")
    

