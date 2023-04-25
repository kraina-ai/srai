# Use cases

`srai` can be used to work with spatial data and use spatial representations in applications on different levels of abstraction. We support (of plan to support) working with pretrained embeddings, training new embeddings, modifying or proposing new embedding methods and more. Some features are not yet available, but we include them here as a roadmap of our plans to make some our our designing decisions more clear.

We present use-cases in a form of user stories obtained using design thinking approach.

## Pre-calculated embeddings

> 🏗️ In progress... 🏗️
>
> Embeddings store is not yet available, as it introduces infrastructure costs to host them. We intend to release pre-trained embeddings from our methods in

As a city planner, I would like to anticipate the amount of traffic on the estate. I work for one of the major cities, for which I found that there are pre-trained embeddings by the `srai` team (or by a member of the community). I want it to run fast and that I don't have to do anything other than download the embeds and a library that can read them.

## Transfer learning

> 🏗️ In progress... 🏗️
>
> Pre-trained model's weights are not yet available for download. We plan to reproduce our experimetns using `srai` and release weights for all of our published models

As an avid urban activist, I would like to be able to build a predictive model based on spatial data for my hometown. I have my data (electric bike charging locations and their use after the pilot program in one estate), but I don't know how to describe the features of the city areas. I cannot find any pre-calculated embeddings for my town. I like the assumption that my area is similar to some other area for which the models exist. I would like to use a model pre-trained on other cities to obtain representations for my city and use them to train my model.

## Embedding training

I work for a logistics company and want to use hex2vec embeddings to predict the demand in given aread of a city. I want them to be based only on our area of operations, so I need to train them myself. I need a simple solution to train my own hex2vec based on OSM data.

## New data source

> 🏗️ In progress... 🏗️
>
> Pre-trained model's weights are not yet available for download. We plan to reproduce our experimetns using `srai` and release weights for all of our published models

I am a planner and I am preparing plans for a new estate. I have a planned road network and distribution of points for it (in accordance with the OSM format) and I want to find out where to put stations for city bikes. I'm fine with using pre-trained model, but my data isn't available anywhere yet. I would like to be able to run predictions on my data using an original highway2vec model.
