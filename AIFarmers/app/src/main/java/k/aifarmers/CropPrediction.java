package k.aifarmers;

import android.content.DialogInterface;
import android.content.Intent;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageButton;
import android.widget.Spinner;
import android.widget.Toast;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;

public class CropPrediction extends AppCompatActivity  {

    String[] months={"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"};
    String cm,loc,url;
    Calendar c = Calendar.getInstance();
    ImageButton s1,s2,s3,s4,s5,s6;
    AlertDialog.Builder builder;

    RequestQueue queue;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        url = new String("http://192.168.20.139:8000/crop");
        queue = Volley.newRequestQueue(this);
        builder = new AlertDialog.Builder(this);

        setContentView(R.layout.activity_crop_prediction);
        cm=months[c.get(Calendar.MONTH)];
        loc="HYA";
        s1=findViewById(R.id.S1);
        s1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                tellme("AL");
            }
        });
        s2=findViewById(R.id.S2);
        s2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                tellme("RD");
            }
        });

        s3=findViewById(R.id.S3);
        s3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                tellme("MN");
            }
        });
        s4=findViewById(R.id.S4);
        s4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                tellme("LT");
            }
        });
        s5=findViewById(R.id.S5);
        s5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                tellme("DK");
            }
        });
        s6=findViewById(R.id.S6);
        s6.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                tellme("DT");
            }
        });

    }
    void tellme(String s)
    {

        HashMap<String,String> hashMap=new HashMap<>();
        hashMap.put("soil",s);
        hashMap.put("month",cm);
        hashMap.put("loc",loc);
        JSONObject jsonObject=new JSONObject(hashMap);
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest
                (Request.Method.POST, url, jsonObject, new Response.Listener<JSONObject>() {

                    @Override
                    public void onResponse(JSONObject response) {
                        Log.d("blah",response.toString());
                        String s="";
                        try {
                            s =response.get("data").toString();
                        }catch (JSONException e){}
                        builder.setMessage(s) .setTitle("Suggested Crops Are : ");

                        //Setting message manually and performing action on button click
                          AlertDialog alert = builder.create();
                        //Setting the title manually
                        alert.setTitle("Suggested Crops Are");
                        alert.show();

                        }





                }, new Response.ErrorListener() {

                    @Override
                    public void onErrorResponse(VolleyError error) {

                    }
                });
        jsonObjectRequest.setRetryPolicy(new DefaultRetryPolicy(20 * 1000, 0,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));


        queue.add(jsonObjectRequest);

    }

}
