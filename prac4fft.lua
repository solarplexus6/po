local cv=require("luacv")

local IMG_WIN = "img"
local IMG_DFT = "dft"
local IMG_IDFT = "inverse dft"
local IMG_DFT_RE = "dft re"
local IMG_DFT_IM = "dft im"
local IMG_MAG = "magnitude"
local IMG_ANGLE = "angle"

function FinalScaleImageInplace(img)
    local m,M=cv.MinMaxLoc(img)
    cv.Scale(img, img, 1.0/(M-m), 1.0*(-m)/(M-m))
end

function LogScaleImageInPlace(img)
    -- Compute log(1 + img)
    cv.AddS( img, cv.ScalarAll(1.0), img, nil );
    cv.Log( img, img);
end

filename=not arg[1] and "land.png" or arg[1]

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
tmpMat2= cv.CreateMat(dft_M,dft_N,cv.CV_64FC1)
tmpMat3= cv.CreateMat(dft_M,dft_N,cv.CV_64FC1)

tmp=cv.CreateMatHeader(dft_M,dft_N,cv.CV_64FC2)

-- copy A to dft_A and pad dft_A with zeros
cv.GetSubRect( dft_A, tmp, cv.Rect(0,0, im.width, im.height));
cv.Copy( complexInput, tmp, nil );
if dft_A.cols > im.width then
  cv.GetSubRect( dft_A, tmp, cv.Rect(im.width,0, dft_A.cols - im.width, im.height));
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
cv.DFT( dft_A, idftC2, cv.CV_DXT_INVERSE, complexInput.height);
idftImage = cv.CreateImage( dft_size, cv.IPL_DEPTH_64F, 1);
cv.Split( idftC2, idftImage, nil, nil, nil);
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
LogScaleImageInPlace(mag)
FinalScaleImageInplace(mag)
cv.ShowImage(IMG_MAG, mag)

-- show dft angle
LogScaleImageInPlace(angle)
FinalScaleImageInplace(angle)
cv.ShowImage(IMG_ANGLE, angle);

-- Rearrange the quadrants of Fourier image so that the origin is at
-- the image center
--ShiftDFT( image_Re);

-- show dft real part
LogScaleImageInPlace(image_Re);
FinalScaleImageInplace(image_Re);
cv.ShowImage(IMG_DFT_RE, image_Re);

-- show dft imaginary part
LogScaleImageInPlace(image_Im);
FinalScaleImageInplace(image_Im);
cv.ShowImage(IMG_DFT_IM, image_Im);

local variants = {
    --1. not changed image
    function ()        
        cv.DFT( dft_A, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print 'not altered idft'
    end,    
    -- 2. im part = 0
    function ()
        cv.Zero(tmpMat)
        cv.Merge(dftRe, tmpMat, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print 'im part = 0'
    end,
    -- 3. re part = 0
    function ()
        cv.Zero(tmpMat)
        cv.Merge(tmpMat, dftIm, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print 'real part = 0'
    end,
    -- 4. norm
    function ()
        cv.CartToPolar(dftRe, dftIm, tmpMat)
        cv.Zero(tmpMat2)
        cv.Merge(tmpMat, tmpMat2, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print 'norm'
    end,
    -- 5. normalized -- obraz widoczny ma byc, bo to kierunki krzywych
    function ()
        -- compute the magnitude
        cv.CartToPolar(dftRe, dftIm, tmpMat)
        cv.Div(dftRe, tmpMat, tmpMat2)
        cv.Div(dftIm, tmpMat, tmpMat3)
        cv.Merge(tmpMat2, tmpMat3, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print 'phase'
    end,
    -- 6. conjugate
    function ()
        cv.ConvertScale(dftIm, tmpMat, -1)
        cv.Merge(dftRe, tmpMat, nil, nil, idftC2);
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print 'conjugate'
    end,
    -- 7. 0.25a+ ib
    function ()
        cv.ConvertScale(dftRe, tmpMat, 0.25)
        cv.Merge(tmpMat, dftIm, nil, nil, idftC2);
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print '0.25a+ ib'
    end,
    -- 8. (a + ib) * (m^2 + n^2)
    function ()
        if scalars == nil then
            scalars = {}
            for i=0,dft_M - 1 do
                for j=0,dft_N - 1 do
                  table.insert(scalars, math.pow(i,2) + math.pow(j,2))
                end
            end
            scalarsMat=cv.Mat(dft_M, dft_N, cv.CV_32FC1, scalars)
        end
                
        cv.Mul(dftRe, scalarsMat, tmpMat)
        cv.Mul(dftIm, scalarsMat, tmpMat2)
        cv.Merge(tmpMat, tmpMat2, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print '(a + ib) * (m^2 + n^2)'
    end,
    -- 9. wyzerujemy gdy nie jest w kwadracie rozmiaru K=50, o środku w (0,0)
    function ()
        cv.Zero(idftC2)
        cv.Zero(tmpMat)
        cv.Zero(tmpMat2)
        local K = 50
        
        CopyCorners(dftRe, tmpMat, K)
        CopyCorners(dftIm, tmpMat2, K)
                
        --cv.CartToPolar(tmpMat, tmpMat2, idftImage)
        --LogScaleImageInPlace(idftImage)
        cv.Merge(tmpMat, tmpMat2, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_INVERSE, complexInput.height)
        cv.Split( idftC2, idftImage, nil, nil, nil)
        print 'filtr dolnoprzepustowy'
    end,
    [0] = function ()
        if translate == nil then
            translate = {}
            for i=0,dft_M - 1 do
                for j=0,dft_N - 1 do
                  table.insert(translate, math.pow(-1, i + j))
                end
            end
            translateMat=cv.Mat(dft_M, dft_N, cv.CV_32FC1, translate)
        end
        cv.Mul(realInput, translateMat, tmpMat)        
        cv.Zero(tmpMat2)
        cv.Merge(tmpMat, tmpMat2, nil, nil, idftC2)
        cv.DFT( idftC2, idftC2, cv.CV_DXT_FORWARD, complexInput.height );
        cv.Split( idftC2, tmpMat, tmpMat2, nil, nil)
        cv.CartToPolar(tmpMat, tmpMat2, idftImage)
        LogScaleImageInPlace(idftImage)
        print 'translacja dft'
    end,
}

function CopyCorners(src, dest, rectSize)
    local rS = rectSize
    local n = src.width
    local m = src.height
    local qstub =cv.CreateMatHeader(rS, rS,cv.CV_64FC1)
    local rstub =cv.CreateMatHeader(rS, rS,cv.CV_64FC1)

    local q = cv.GetSubRect(src,qstub,cv.Rect(0,0, rS,rS))
    local r = cv.GetSubRect(dest,rstub,cv.Rect(0,0,rS,rS))
    cv.Copy(q, r)
    q = cv.GetSubRect(src,qstub,cv.Rect(n - rS, m - rS, rS, rS))
    r = cv.GetSubRect(dest,rstub,cv.Rect(n - rS, m - rS, rS, rS))
    cv.Copy(q, r)
    q = cv.GetSubRect(src,qstub,cv.Rect(0, m - rS, rS, rS))
    r = cv.GetSubRect(dest,rstub,cv.Rect(0, m - rS, rS, rS))
    cv.Copy(q, r)
    q = cv.GetSubRect(src,qstub,cv.Rect(n - rS, 0, rS, rS))
    r = cv.GetSubRect(dest,rstub,cv.Rect(n - rS, 0, rS, rS))
    cv.Copy(q, r)
end

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
    --LogScaleImageInPlace(idftImage)
    FinalScaleImageInplace(idftImage)
    cv.ShowImage(IMG_IDFT, idftImage)
end
cv.DestroyAllWindows()