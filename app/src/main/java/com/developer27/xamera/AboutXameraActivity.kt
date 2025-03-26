package com.developer27.xamera

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.developer27.xamera.databinding.ActivityAboutXameraBinding

class AboutXameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAboutXameraBinding
    private var umLogoClickCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAboutXameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // When the title container is clicked, open the URL in a browser.
        binding.titleContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        // When the UM logo is clicked 5 times, show a toast.
        binding.umLogo.setOnClickListener {
            val url = "https://umich.edu/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }
    }
}