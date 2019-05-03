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

    constructor(private location: Location, private router: Router,
                iconRegistry: MatIconRegistry, sanitizer: DomSanitizer,
    ) {
        iconRegistry.addSvgIcon('back',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/back.svg'));

        iconRegistry.addSvgIcon('forward',
            sanitizer.bypassSecurityTrustResourceUrl('assets/img/forward.svg'));
    }

    getAnimationData(outlet: RouterOutlet) {
        return outlet && outlet.activatedRouteData && outlet.activatedRouteData['animation'];
    }

    go_back(): void {
        this.location.back();
    }

    go_forward(): void {
        this.location.forward();
    }
}
