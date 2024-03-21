import React, { useState, useEffect } from "react";
import Grid from "@material-ui/core/Grid";
import { makeStyles } from "@material-ui/core/styles";
import Typography from "@material-ui/core/Typography";
import TextField from "@material-ui/core/TextField";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Checkbox from "@material-ui/core/Checkbox";
import Select from "@material-ui/core/Select";
import InputLabel from "@material-ui/core/InputLabel";
import MenuItem from "@material-ui/core/MenuItem";
import FormControl from "@material-ui/core/FormControl";
import Box from "@material-ui/core/Box";
import Button from "@material-ui/core/Button";
import FormGroup from "@material-ui/core/FormGroup";
import axios from "axios";
import Snackbar from "@material-ui/core/Snackbar";
import MuiAlert from "@material-ui/lab/Alert";
import sha256 from "sha256";
import initJWTService from "jwt-service";
import { CircularProgress } from "@material-ui/core";
import SavedPatientDetails from "../SavedPatientDetails/SavedPatientDetails";

function Alert(props) {
  return <MuiAlert elevation={6} variant="filled" {...props} />;
}

const useStyles = makeStyles((theme) => ({
  formControl: {
    margin: theme.spacing(1),
    minWidth: 600,
  },
  // Add additional styles as needed
}));

const YourComponent = () => {
  const classes = useStyles();

  // Define state variables using useState
  const [exampleState, setExampleState] = useState(initialValue);

  // Define useEffect for any side effects
  useEffect(() => {
    // Perform side effects here
    return () => {
      // Cleanup side effects here if necessary
    };
  }, []); // Empty dependency array means useEffect runs only once on component mount

  // Define your component JSX
  return (
    <div>
      {/* Your JSX code here */}
    </div>
  );
};

export default YourComponent;
