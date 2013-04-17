-- Histogram equalization
function love.load()
    -- image = love.graphics.newImage("images/lena.jpg")
    origSource = love.image.newImageData("images/lena.jpg")
    --origSource = love.image.newImageData("images/moo2heq1.jpg")
    --origSource = love.image.newImageData("images/test.png")
    --origSource = love.image.newImageData("images/Unequalized_Hawkes_Bay_NZ.jpg")
    local source = love.image.newImageData(origSource:getWidth(), origSource:getHeight())
    --local source = origSource
    source:paste(origSource, 0, 0)

    origImage = love.graphics.newImage(source)

    size = source:getWidth()*source:getHeight()
    range = 256 - 1

    alpha = 0.5 -- dla czesciowego wyrownywania

    local hist = {}
    local maxIntensity = 255
    local minIntensity = 0
    source:mapPixel(function ( x, y, r, g, b, a )
            local intensity = r + g + b
            local old = hist[intensity] or 0
            hist[intensity] = old + 1

            local mapped = math.floor(intensity/3)
            minIntensity = mapped < minIntensity and mapped or minIntensity
            maxIntensity = mapped > maxIntensity and mapped or maxIntensity
        return r, g, b, a
    end)
    cdf = {}
    cdfMin = range

    for k,v in pairs(hist) do
        local sum = 0
        for i=0,k do
            if (hist[i]) then
                sum = sum + hist[i]/size
            end
        end
        local result = sum
        cdf[k] = result


        if (result < cdfMin) then
            cdfMin = result
        end
    end
    source:mapPixel(function (x, y, r, g, b, a)
        --local h = math.floor( cdf[r+g+b] * (maxIntensity - minIntensity) + minIntensity)
        local h = math.floor( alpha* cdf[r+g+b] * range + (1-alpha) * (r+g+b)/3)
        return h, h, h, a
    end)
    image = love.graphics.newImage(source)
end

function love.draw()    
    love.graphics.draw(image, 0, 0, 0, 0.5, 0.5)
    love.graphics.draw(origImage, 500, 0, 0, 0.5, 0.5)
end

function math.clamp(low, n, high) return math.min(math.max(n, low), high) end

function love.update()
    local source = love.image.newImageData(origSource:getWidth(), origSource:getHeight())
    source:paste(origSource, 0, 0)
    source:mapPixel(function (x, y, r, g, b, a)
        --local h = math.floor( cdf[r+g+b] * (maxIntensity - minIntensity) + minIntensity)
        local intensitySum = r + g + b
        local h = math.floor( alpha * cdf[r+g+b] * range + (1-alpha) * intensitySum/3)
        scale = h/(intensitySum/3)
        local newR = math.clamp(0, math.floor(r * scale), 255)
        local newG = math.clamp(0, math.floor(g * scale), 255)
        local newB = math.clamp(0, math.floor(b * scale), 255)
        return newR, newG, newB, a
    end)
    image = love.graphics.newImage(source)
end

function love.keypressed(key, unicode)
    if (key == 'up') then
        alpha = alpha + 0.1
    elseif (key == 'down') then
        alpha = alpha - 0.1
    end
end