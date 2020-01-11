package ece.cpen502;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.util.List;

    public class XYLineChart_AWT extends ApplicationFrame {

        public XYLineChart_AWT( String applicationTitle, String chartTitle, List<Double> epochErrors) {
            super(applicationTitle);

            final XYSeries errors = new XYSeries( "" );
            for (int i = 0; i < epochErrors.size(); i++) {
                errors.add( i + 1 , epochErrors.get(i));
            }

            XYSeriesCollection dataset = new XYSeriesCollection( );
            dataset.addSeries(errors);

            JFreeChart xylineChart = ChartFactory.createXYLineChart(
                    chartTitle ,
                    "Hidden neurons" ,
                    "RMS Error" ,
                    dataset,
                    PlotOrientation.VERTICAL ,
                    true , true , false);

            ChartPanel chartPanel = new ChartPanel( xylineChart );
            chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
            final XYPlot plot = xylineChart.getXYPlot( );

            XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer( );
            renderer.setSeriesPaint( 0 , Color.BLUE);
            renderer.setSeriesStroke( 0 , new BasicStroke( 1.0f ) );
            renderer.setSeriesShapesVisible(0, false);

            plot.setRenderer( renderer );
            setContentPane( chartPanel );
        }


}
