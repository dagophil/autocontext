"""
Author: DerThorsten
"""
import numpy
from numbers import Number


class BlockWithMargin(object):
    def __init__(self, block, margin):
        shape = block.shape
        dim = len(shape)
        self.innerBlock = block
        if(isinstance(margin, Number)):
            m = [margin]*dim
        else:
            m = margin

        begin = list(block.begin)
        end = list(block.end)


        localInnerBegin = [None]*dim
        localInnerEnd = [None]*dim
        for d in range(dim):
            begin[d] = max(0, int(begin[d])-m[d])
            end[d] = min(shape[d], end[d]+m[d])

            localInnerBegin[d] = block.begin[d] - begin[d]
            localInnerEnd[d] = block.end[d] - begin[d]

        self.localInnerBlock = Block(localInnerBegin,localInnerEnd, block.blocking)
        self.outerBlock = Block(begin,end, block.blocking)

    def __str__(self):
        return "[InnerBlock"+str(self.innerBlock)+" OuterBlock"+str(self.outerBlock)+"]"


class Block(object):
    def __init__(self,begin, end, blocking):
        self.begin = begin
        self.end = end
        self.slicing = []
        self.blocking = blocking
        for b,e in zip(self.begin, self.end):
            self.slicing.append(slice(b, e))


    def blockWithMargin(self, margin):
        return BlockWithMargin(block=self,margin=margin)

    @property
    def shape(self):
        return self.blocking.shape


    def __str__(self):
        return "["+str(tuple(self.begin))+" - "+str(tuple(self.end))+"]"




class Blocking(object):
    def __init__(self,shape,blockShape=None, roi=None ):
        self.shape = shape
        self.nDim = len(shape)


        assert roi is None
        self.blockShape = None
        nDim = len(shape)
        if blockShape is None:
            self.blockShape = [64]*nDim
        elif isinstance(blockShape,Number):
            self.blockShape = [int(blockShape)]*nDim
        else:
            self.blockShape = list(blockShape)

        for d in range(nDim):
            self.blockShape[d] = min(self.blockShape[d], shape[d])



        ns = numpy.array(self.shape)
        nbs = numpy.array(self.blockShape)

        self.blocksPerAxis = numpy.ceil(ns.astype('float')/nbs).astype('int')

        nBlocks = 1
        for nb in self.blocksPerAxis:
            nBlocks*=nb
        self.nBlocks = nBlocks



    def yieldBlocks(self):
        blocksPerAxis = self.blocksPerAxis
        shape = self.shape
        bshape = self.blockShape

        if self.nDim == 2 :

            for bx in range(blocksPerAxis[0]):
                for by in range(blocksPerAxis[1]):

                    bs = (bx*bshape[0],by*bshape[1])
                    be = (min((bx+1)*bshape[0],shape[0]),
                          min((by+1)*bshape[1],shape[1]))

                    yield Block(bs,be,self)

        elif self.nDim == 3 :
            for bx in range(blocksPerAxis[0]):
                for by in range(blocksPerAxis[1]):
                    for bz in range(blocksPerAxis[2]):
                        bs = (bx*bshape[0],by*bshape[1],bz*bshape[2])
                        be = (min((bx+1)*bshape[0],shape[0]),
                              min((by+1)*bshape[1],shape[1]),
                              min((bz+1)*bshape[2],shape[2]))

                        yield Block(bs,be,self)

        elif self.nDim == 4 :
            for bx in range(blocksPerAxis[0]):
                for by in range(blocksPerAxis[1]):
                    for bz in range(blocksPerAxis[2]):
                        for bt in range(blocksPerAxis[3]):
                            bs = (bx*bshape[0],by*bshape[1],bz*bshape[2],bt*bshape[3])
                            be = (min((bx+1)*bshape[0],shape[0]),
                                  min((by+1)*bshape[1],shape[1]),
                                  min((bz+1)*bshape[2],shape[2]),
                                  min((bt+1)*bshape[3],shape[3]))

                            yield Block(bs,be,self)

        elif self.nDim == 5 :
            for bx in range(blocksPerAxis[0]):
                for by in range(blocksPerAxis[1]):
                    for bz in range(blocksPerAxis[2]):
                        for bt in range(blocksPerAxis[3]):
                            for bc in range(blocksPerAxis[4]):
                                bs = (bx*bshape[0],by*bshape[1],bz*bshape[2],bt*bshape[3],bc*bshape[4])
                                be = (min((bx+1)*bshape[0],shape[0]),
                                      min((by+1)*bshape[1],shape[1]),
                                      min((bz+1)*bshape[2],shape[2]),
                                      min((bt+1)*bshape[3],shape[3]),
                                      min((bc+1)*bshape[4],shape[4]))

                                yield Block(bs,be,self)

        else:
            raise Exception("%d dimensional data not supported." % self.nDim)


if __name__ == "__main__":

    for block in Blocking([1,10,1],[1,3,2]).yieldBlocks():
        print block

