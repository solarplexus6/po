local cv=require("luacv")

local IMG_WIN = "img"
local IMG_IDFT = "inverse dft"
local IMG_DFT_RE = "dft re"
local IMG_DFT_IM = "dft im"
local IMG_MAG = "magnitude"
local IMG_ANGLE = "angle"

function FinalScaleImageInplace(img)
    local m,M=cv.MinMaxLoc(img)
    cv.Scale(img, img, 1.0/(M-m), 1.0*(-m)/(M-m))
end

function LogScaleImageInPlance(img)
    -- Compute log(1 + img)
    cv.AddS( img, cv.ScalarAll(1.0), img, nil );
    cv.Log( img, img);
end

function ShiftDFT(src_im)
    local center=cv.Point(src_im.height/2,src_im.width/2)
    local qstub={} 
    for i=1,4 do
        qstub[i]=cv.CreateMatHeader(src_im.height,src_im.width,cv.CV_64FC1)
    end

    local q1=cv.GetSubRect(src_im,qstub[1],cv.Rect(0,0,center.x,center.y))
    local q2=cv.GetSubRect(src_im,qstub[2],cv.Rect(center.x,0,center.x,center.y))
    local q3=cv.GetSubRect(src_im,qstub[3],cv.Rect(center.x,center.y,center.x,center.y))
    local q4=cv.GetSubRect(src_im,qstub[4],cv.Rect(0,center.y,center.x,center.y))
    
    local tmp=cv.CreateMat(src_im.height/2,src_im.width/2,cv.CV_64FC1)

    cv.Copy(q3,tmp)
    cv.Copy(q1,q3)
    cv.Copy(tmp,q1)
    cv.Copy(q4,tmp)
    cv.Copy(q2,q4)
    cv.Copy(tmp,q2)
end

filename=not arg[1] and "lena.png" or arg[1]

im = cv.LoadImage( filename, cv.CV_LOAD_IMAGE_GRAYSCALE );
if not im then error("Couldn't load image") end

im_size=cv.GetSize(im)
realInput = cv.CreateImage( im_size,cv.IPL_DEPTH_64F, 1);
imaginaryInput = cv.CreateImage( im_size, cv.IPL_DEPTH_64F, 1);
complexInput = cv.CreateImage( im_size, cv.IPL_DEPTH_64F, 2);

cv.Scale(im, realInput, 1.0, 0.0);
cv.Zero(imaginaryInput);
cv.Merge(realInput, imaginaryInput, nil, nil, complexInput);

dft_M = cv.GetOptimalDFTSize( im.height - 1 );
dft_N = cv.GetOptimalDFTSize( im.width - 1 );

dft_A = cv.CreateMat( dft_M, dft_N, cv.CV_64FC2 );
dft_size=cv.Size(dft_N, dft_M)
image_Re = cv.CreateImage( dft_size, cv.IPL_DEPTH_64F, 1);
image_Im = cv.CreateImage( dft_size, cv.IPL_DEPTH_64F, 1);
dftRe = cv.CreateMat( dft_M, dft_N, cv.CV_64FC1 );
dftIm = cv.CreateMat( dft_M, dft_N, cv.CV_64FC1 );
tmpMat = cv.CreateMat( dft_M, dft_N, cv.CV_64FC1 );

tmp=cv.CreateMatHeader(dft_M,dft_N,cv.CV_64FC2)

-- copy A to dft_A and pad dft_A with zeros
cv.GetSubRect( dft_A, tmp, cv.Rect(0,0, im.width, im.height));
cv.Copy( complexInput, tmp, nil );
if dft_A.cols > im.width then
  cv.GetSubRect( dft_A, tmp, cvRect(im.width,0, dft_A.cols - im.width, im.height));
  cv.Zero(tmp);
end

-- no need to pad bottom part of dft_A with zeros because of
-- use nonzero_rows parameter in cvDFT() call below

cv.DFT( dft_A, dft_A, cv.CV_DXT_FORWARD, complexInput.height );

cv.NamedWindow(IMG_WIN, 0);
cv.NamedWindow(IMG_DFT_RE, 0);
cv.NamedWindow(IMG_DFT_IM, 0);
cv.NamedWindow(IMG_IDFT, 0);
cv.NamedWindow(IMG_MAG, 0);
cv.NamedWindow(IMG_ANGLE, 0);
cv.ShowImage(IMG_WIN, im);

-- Split Fourier in real and imaginary parts
cv.Split( dft_A, image_Re, image_Im, nil, nil );
cv.Split( dft_A, dftRe, dftIm, nil, nil);

idftC2 = cv.CreateMat( dft_M, dft_N, cv.CV_64FC2 );
idftC1 = cv.CreateMat( dft_M, dft_N, cv.CV_64FC1 );
cv.DFT( dft_A, idftC2, cv.CV_DXT_INVERSE, complexInput.height);
idftImage = cv.CreateImage( dft_size, cv.IPL_DEPTH_64F, 1);
cv.Split( idftC2, idftImage, nil, nil, nil);
LogScaleImageInPlance(idftImage)
FinalScaleImageInplace(idftImage)
cv.ShowImage(IMG_IDFT, idftImage)

mag = cv.CreateImage( dft_size, cv.IPL_DEPTH_64F, 1);
angle = cv.CreateImage( dft_size, cv.IPL_DEPTH_64F, 1);
-- Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
--cv.Pow( image_Re, mag, 2.0);
--cv.Pow( image_Im, magTmp, 2.0);
--cv.Add( mag, magTmp, mag, nil);
--cv.Pow( mag, mag, 0.5 );
cv.CartToPolar(image_Re, image_Im, mag, angle)

-- show dft magnitude
LogScaleImageInPlance(mag)
FinalScaleImageInplace(mag)
cv.ShowImage(IMG_MAG, mag)

-- show dft angle
LogScaleImageInPlance(angle)
FinalScaleImageInplace(angle)
cv.ShowImage(IMG_ANGLE, angle);

-- Rearrange the quadrants of Fourier image so that the origin is at
-- the image center
--ShiftDFT( image_Re);

-- show dft real part
LogScaleImageInPlance(image_Re);
FinalScaleImageInplace(image_Re);
cv.ShowImage(IMG_DFT_RE, image_Re);

-- show dft imaginary part
LogScaleImageInPlance(image_Im);
FinalScaleImageInplace(image_Im);
cv.ShowImage(IMG_DFT_IM, image_Im);

local variants = {
    --1. not changed image
    function ()        
        cv.DFT( dft_A, idftC2, cv.CV_DXT_INVERSE, complexInput.height) 
        cv.Split( idftC2, idftImage, nil, nil, nil)
    end,
    -- 2. real part
    function ()        
        cv.DFT( dftRe, idftC1, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC1, idftImage, nil, nil, nil)
    end,
    -- 3. imaginary part
    function ()        
        cv.DFT( dftIm, idftC1, cv.CV_DXT_INVERSE, complexInput.height);            
        cv.Split( idftC1, idftImage, nil, nil, nil);
    end,
    -- 4. real part = 0
    function ()
        cv.Zero(tmpMat)
        cv.Merge(dftRe, tmpMat, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil);
    end,
    -- 5. im part = 0
    function ()
        cv.Zero(tmpMat)
        cv.Merge(tmpMat, dftIm, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil);
    end
}
running = true
while running do        
    local key = cv.WaitKey(-1);
    if key == 27 then
        break
    end
    local variant = variants[key - string.byte('0')]
    if variant ~= nil then
        variant()
    end
    --LogScaleImageInPlance(idftImage)
    FinalScaleImageInplace(idftImage)
    cv.ShowImage(IMG_IDFT, idftImage)
end
cv.DestroyAllWindows()