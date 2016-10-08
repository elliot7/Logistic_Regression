class logistic_regression
{
	//double theta[][] = new double[4][1];
	double hypo[][];
	double cost[][];
	double learning_rate = 0.1;//old_vale=0.1;
	double iterations = 1000000;
	
	// Sigmoid Function for Hypothesis
	
	public double[][] sigmoid(double[][] feature,double[][] weights)
	{
		double[][] z = new double[feature.length][weights[0].length];
		double sum = 0;
		for(int i=0;i<feature.length;i++)
		{
			for(int j=0;j<weights[0].length;j++)
			{
				for(int k=0;k<weights.length;k++)
				{
					sum = sum + feature[i][k] * weights[k][j];
				}
				z[i][j] = sum;
				sum=0;
			}
		}
		hypo = new double[z.length][z[0].length];
		for(int i=0;i<z.length;i++)
		{
			for(int j=0;j<z[0].length;j++)
			{
				hypo[i][j] = (1/( 1+( Math.exp( (-1)*z[i][j] ) ) ) );
			}
		}
		return hypo;
	}
	
	// Cost Function J
	
	public double[][] cost(double[][] feature_value,double[][] parameters,double[][] y)
	{
		double sum=0;
		double[][] transpose_y = new double[y[0].length][y.length];
		for(int i=0;i<y.length;i++)
		{
			for(int j=0;j<y[0].length;j++)
			{
				transpose_y[j][i] = (-1)*y[i][j];
			}
		}
		double[][] transpose_y_minus = new double[y[0].length][y.length];
		for(int i=0;i<y.length;i++)
		{
			for(int j=0;j<y[0].length;j++)
			{
				transpose_y_minus[j][i] = (1-y[i][j]);
			}
		}
		double[][] hypothe = new double[feature_value.length][1];
		hypothe = sigmoid(feature_value,parameters);
		double[][] log_hypothe = new double[feature_value.length][1];
		for(int i=0;i<hypothe.length;i++)
		{
			for(int j=0;j<hypothe[0].length;j++)
			{
				log_hypothe[i][j] = Math.log(hypothe[i][j]);
			}
		}
		double[][] log_hypothe_minus = new double[feature_value.length][1];
		for(int i=0;i<log_hypothe_minus.length;i++)
		{
			for(int j=0;j<log_hypothe_minus[0].length;j++)
			{
				log_hypothe_minus[i][j] = Math.log(1-hypothe[i][j]);
			}
		}
		double[][] a = new double[1][1];
		double[][] b = new double[1][1];
		for(int i=0;i<transpose_y.length;i++)
		{
			for(int j=0;j<log_hypothe[0].length;j++)
			{
				for(int k=0;k<log_hypothe.length;k++)
				{
					sum = sum + transpose_y[i][k] * log_hypothe[k][j];
				}
				a[i][j] = sum;
				sum=0;
			}
		}
		for(int i=0;i<transpose_y_minus.length;i++)
		{
			for(int j=0;j<log_hypothe_minus[0].length;j++)
			{
				for(int k=0;k<log_hypothe_minus.length;k++)
				{
					sum = sum + transpose_y_minus[i][k] * log_hypothe_minus[k][j];
				}
				b[i][j] = sum;
				sum=0;
			}
		}
		cost = new double[1][1];
		for(int i=0;i<cost.length;i++)
		{
			for(int j=0;j<cost[0].length;j++)
			{
				cost[i][j] = ((a[i][j]-b[i][j])/feature_value.length);
			}
		}
		
			return cost;
	}
	
	// Gradient Descent for parameter optimization
	
	//Gradient Calculation
	public double[][] gradient(double[][] feature_val, double[][] theta,double[][] prediction)
	{
		double sum=0;
		double[][] hypothesis = new double[feature_val.length][theta[0].length];
		double[][] hypo_minus_predict = new double[hypothesis.length][hypothesis[0].length];
		double[][] feature_val_transpose = new double[feature_val[0].length][feature_val.length];
		double[][] secondterm_product = new double[theta.length][1];
		// double[][] grad = new double[theta.length][1];
				hypothesis = sigmoid(feature_val,theta);
		
		for(int i=0;i<hypothesis.length;i++)
		{
			for(int j=0;j<hypothesis[0].length;j++)
			{
				hypo_minus_predict[i][j] = (hypothesis[i][j] - prediction[i][j]); 
			}
		}
		
		for(int i=0;i<feature_val.length;i++)
		{
			for(int j=0;j<feature_val[0].length;j++)
			{
				feature_val_transpose[j][i] = feature_val[i][j];
			}
		}		
		
		for(int i=0;i<feature_val_transpose.length;i++)
		{
			for(int j=0;j<hypo_minus_predict[0].length;j++)
			{
				for(int k=0;k<hypo_minus_predict.length;k++)
				{
					sum = sum + feature_val_transpose[i][k] * hypo_minus_predict[k][j];
				}
				secondterm_product[i][j] = sum;
				sum=0;
			}
		}
		
				
		
		return secondterm_product;
}

	
	//Parameter Update
	
	public double[][] parameter_update(double[][] features_val,double[][] parameter_val,double[][] label)
	{
		// double[][] updated_theta = new double[parameter_val.length][parameter_val[0].length];
		double[][] secondpart = new double[parameter_val.length][parameter_val[0].length];
		double[][] cost_function = new double[label[0].length][features_val[0].length];
		for(int k=0;k<iterations;k++)
		{
			secondpart = gradient(features_val,parameter_val,label);
			for(int i=0;i<parameter_val.length;i++)
			{
				for(int j=0;j<parameter_val[0].length;j++)
				{
					parameter_val[i][j] = parameter_val[i][j] - (((learning_rate/features_val.length) * secondpart[i][j]));
				}
			}
			
			
			for(int i=0;i<1;i++)
			{
				for(int j=0;j<1;j++)
				{
					cost_function = cost(features_val,parameter_val,label);
					System.out.println(cost_function[i][j]);
				}
			}
			
			
	
		}
		
		return parameter_val;
	}
	
	//Classifier
	public double[][] classify(double[][] feature_values,double[][] weights)
	{
		double[][] predicted_value = new double[feature_values.length][weights[0].length];
		predicted_value = sigmoid(feature_values,weights);
		return predicted_value;
	}
}
