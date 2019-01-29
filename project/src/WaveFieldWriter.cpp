#include "WaveFieldWriter.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <limits>

#include "basisfunctions.h"
#include "GEMM.h"

WaveFieldWriter::WaveFieldWriter(std::string const& baseName, GlobalConstants const& globals, LocalConstants const& locals, double interval, int pointsPerDim)
  : m_rank(locals.rank), m_step(0), m_interval(interval), m_lastTime(-std::numeric_limits<double>::max()), m_pointsPerDim(pointsPerDim)
{  
  if (!baseName.empty()) {
    if (m_rank == 0) {
      m_xdmf.open((baseName + ".xdmf").c_str());
      m_xdmf << "<?xml version=\"1.0\" ?>" << std::endl
             << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">" << std::endl
             << "<Xdmf Version=\"2.0\">" << std::endl
             << "  <Domain>" << std::endl
             << "    <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"" << m_pointsPerDim * globals.Y << " " << m_pointsPerDim * globals.X << "\"/>" << std::endl
             << "    <Geometry GeometryType=\"ORIGIN_DXDY\">" << std::endl
             << "      <DataItem Format=\"XML\" Dimensions=\"2\">0.0 0.0</DataItem>" << std::endl
             << "      <DataItem Format=\"XML\" Dimensions=\"2\">" << globals.hy / m_pointsPerDim << " " << globals.hx / m_pointsPerDim << "</DataItem>" << std::endl
             << "    </Geometry>" << std::endl
             << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;
    }

    int gridSize = m_pointsPerDim * m_pointsPerDim * locals.elts_size[0] * locals.elts_size[1];
    m_pressure = new float[gridSize];
    m_uvel = new float[gridSize];
    m_vvel = new float[gridSize];
    
    std::size_t lastFound = 0;
    std::size_t found;
    while ((found = baseName.find("/", lastFound+1)) != std::string::npos) {
      lastFound = found;
    }
    if (lastFound > 0) {
      ++lastFound;
    }
    m_dirName = baseName.substr(0, lastFound);
    m_baseName = baseName.substr(lastFound);
    
    unsigned subGridSize = m_pointsPerDim * m_pointsPerDim;
    m_subsampleMatrix = new double[subGridSize * NUMBER_OF_BASIS_FUNCTIONS];
    double subGridSpacing = 1.0 / (m_pointsPerDim + 1);
    for (int bf = 0; bf < NUMBER_OF_BASIS_FUNCTIONS; ++bf) {
      for (int y = 0; y < m_pointsPerDim; ++y) {
        for (int x = 0; x < m_pointsPerDim; ++x) {
          double xi = (x+1) * subGridSpacing;
          double eta = (y+1) * subGridSpacing;
          m_subsampleMatrix[bf * subGridSize + (y * m_pointsPerDim + x)] = (*basisFunctions[bf])(xi, eta);
        }
      }
    }
    m_subsamples = new double[subGridSize * NUMBER_OF_QUANTITIES];
    memset(m_subsamples, 0, subGridSize * NUMBER_OF_QUANTITIES * sizeof(double));
  }
}

WaveFieldWriter::~WaveFieldWriter()
{
  if (!m_baseName.empty()) {
    if (m_rank == 0) {
      m_xdmf << "    </Grid>" << std::endl
             << "  </Domain>" << std::endl
             << "</Xdmf>" << std::endl;
      m_xdmf.close();
    }

    delete[] m_subsamples;
    delete[] m_subsampleMatrix;
    delete[] m_pressure;
    delete[] m_uvel;
    delete[] m_vvel;
  }
}

void WaveFieldWriter::writeTimestep(double time, Grid<DegreesOfFreedom>& degreesOfFreedomGrid, GlobalConstants const& globals, LocalConstants const& locals, bool forceWrite)
{
  if (!m_baseName.empty() && (time >= m_lastTime + m_interval || forceWrite)) {
    m_lastTime = time;
    
    std::stringstream pressureFileName, uvelFileName, vvelFileName;
    pressureFileName << m_baseName << "_pressure" << m_step << ".bin";
    uvelFileName << m_baseName << "_u" << m_step << ".bin";
    vvelFileName << m_baseName << "_v" << m_step << ".bin";
    
    if (m_rank == 0) {
      m_xdmf << "      <Grid Name=\"step_" << m_step << "\" GridType=\"Uniform\">" << std::setw(0) << std::endl
             << "        <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>" << std::endl
             << "        <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>" << std::endl
             << "        <Time Value=\"" << time << "\"/>" << std::endl
             << "        <Attribute Name=\"pressure\" Center=\"Node\">" << std::endl
             << "          <DataItem Format=\"Binary\" DataType=\"Float\" Precision=\"4\" Dimensions=\"" << m_pointsPerDim * globals.Y << " " << m_pointsPerDim * globals.X << "\">" << std::endl
             << "            " << pressureFileName.str() << std::endl
             << "          </DataItem>" << std::endl
             << "        </Attribute>" << std::endl
             << "        <Attribute Name=\"u\" Center=\"Node\">" << std::endl
             << "          <DataItem Format=\"Binary\" DataType=\"Float\" Precision=\"4\" Dimensions=\"" << m_pointsPerDim * globals.Y << " " << m_pointsPerDim * globals.X << "\">" << std::endl
             << "            " << uvelFileName.str() << std::endl
             << "          </DataItem>" << std::endl
             << "       </Attribute>" << std::endl
             << "        <Attribute Name=\"v\" Center=\"Node\">" << std::endl
             << "          <DataItem Format=\"Binary\" DataType=\"Float\" Precision=\"4\" Dimensions=\"" << m_pointsPerDim * globals.Y << " " << m_pointsPerDim * globals.X << "\">" << std::endl
             << "            " << vvelFileName.str() << std::endl
             << "          </DataItem>" << std::endl
             << "        </Attribute>" << std::endl
             << "      </Grid>" << std::endl;
    }

    unsigned subGridSize = m_pointsPerDim * m_pointsPerDim;
    for (int y = 0; y < degreesOfFreedomGrid.Y(); ++y) {
      for (int x = 0; x < degreesOfFreedomGrid.X(); ++x) {
        DGEMM(  subGridSize, NUMBER_OF_QUANTITIES, NUMBER_OF_BASIS_FUNCTIONS,
                1.0, m_subsampleMatrix, subGridSize,
                degreesOfFreedomGrid.get(x, y), NUMBER_OF_BASIS_FUNCTIONS,
                0.0, m_subsamples, subGridSize );

        for (int ysub = 0; ysub < m_pointsPerDim; ++ysub) {
          for (int xsub = 0; xsub < m_pointsPerDim; ++xsub) {
            unsigned subIndex = ysub * m_pointsPerDim + xsub;
            unsigned targetIndex = (y*m_pointsPerDim+ysub)*m_pointsPerDim*degreesOfFreedomGrid.X() + (x*m_pointsPerDim+xsub);
            m_pressure[targetIndex] = m_subsamples[0 * subGridSize + subIndex];
            m_uvel[targetIndex] = m_subsamples[1 * subGridSize + subIndex];
            m_vvel[targetIndex] = m_subsamples[2 * subGridSize + subIndex];
          }
        }
      }
    }
    
    MPI_File pressureFile, uFile, vFile;
    MPI_Status status;
    
    MPI_File_open(MPI_COMM_WORLD, (m_dirName + pressureFileName.str()).c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &pressureFile);
    MPI_File_open(MPI_COMM_WORLD, (m_dirName + uvelFileName.str()).c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &uFile);
    MPI_File_open(MPI_COMM_WORLD, (m_dirName + vvelFileName.str()).c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &vFile);
    
    int gridStartIndex = locals.start_elts[1]*subGridSize*globals.X + locals.start_elts[0]*m_pointsPerDim;
    for (int y = 0; y < locals.elts_size[1] * m_pointsPerDim; y++) {
      int viewStartIndex = gridStartIndex + y*m_pointsPerDim*globals.X;
      
      MPI_File_write_at(pressureFile, viewStartIndex * sizeof(float), m_pressure + y*m_pointsPerDim*locals.elts_size[0], m_pointsPerDim*locals.elts_size[0], MPI_FLOAT, &status);
      MPI_File_write_at(uFile, viewStartIndex * sizeof(float), m_uvel + y*m_pointsPerDim*locals.elts_size[0], m_pointsPerDim*locals.elts_size[0], MPI_FLOAT, &status);
      MPI_File_write_at(vFile, viewStartIndex * sizeof(float), m_vvel + y*m_pointsPerDim*locals.elts_size[0], m_pointsPerDim*locals.elts_size[0], MPI_FLOAT, &status);
    }
    
    MPI_File_close(&pressureFile);
    MPI_File_close(&uFile);    
    MPI_File_close(&vFile);
    
    ++m_step;
  }
}
