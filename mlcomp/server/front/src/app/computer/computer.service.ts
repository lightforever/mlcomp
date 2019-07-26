import {Injectable} from '@angular/core';
import {BaseService} from "../base.service";

@Injectable({
    providedIn: 'root'
})
export class ComputerService extends BaseService {
    protected collection_part: string = 'computers';
    protected single_part: string = 'computer';


}