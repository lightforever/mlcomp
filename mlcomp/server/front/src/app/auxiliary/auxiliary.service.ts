import {Injectable} from '@angular/core';
import {BaseService} from "../base.service";

@Injectable({
    providedIn: 'root'
})
export class AuxiliaryService extends BaseService {
    protected collection_part: string;
    protected single_part: string = 'auxiliary';


}