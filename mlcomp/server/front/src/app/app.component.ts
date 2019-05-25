import {Component} from '@angular/core';
import {Router, RouterOutlet} from "@angular/router";
import {slideInAnimation} from './animations';
import {Location} from "@angular/common";
import {MatIconRegistry} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css'],
    animations: [slideInAnimation]
})
export class AppComponent {
    title = 'ML comp dashboard';
}
